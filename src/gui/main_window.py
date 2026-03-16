import sys
import os
import webbrowser
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox,
    QProgressBar, QStatusBar, QMessageBox, QFileDialog, QGroupBox,
    QFormLayout, QSpinBox, QCheckBox, QListWidget, QSplitter,
    QLineEdit, QProgressDialog, QSystemTrayIcon, QStyle, QScrollArea,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTranslator, QLocale, QTimer
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QTextCursor
import shutil
import gc
import traceback

# Import core modules
from core.hardware_check import HardwareCheck
from core.model_manager import ModelManager
from core.data_loader import DataLoader
from core.trainer import ModelTrainer
from core.inference import InferenceEngine
from core.exporter import ModelExporter
from integration.ollama_bridge import OllamaBridge
from utils.logger import Logger
from utils.config import Config
from utils import integrity
from utils import memory
from utils import deps

class Worker(QThread):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(float)
    progress_pct = pyqtSignal(float, float)  # pct, eta
    success = pyqtSignal(object)
    failure = pyqtSignal(str)

    def __init__(self, fn, *args, expects_progress=False, expects_value=False, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.expects_progress = expects_progress
        self.expects_value = expects_value

    def run(self):
        try:
            progress_cb = None
            value_cb = None
            if self.expects_progress:
                def _prog(pct, eta=None):
                    self.progress_pct.emit(pct, eta if eta is not None else -1)
                progress_cb = _prog
            if self.expects_value:
                def _val(val):
                    self.progress_value.emit(val)
                value_cb = _val
            kwargs = dict(self.kwargs)
            if progress_cb:
                kwargs["progress_callback"] = progress_cb
            if value_cb:
                kwargs["progress_value"] = value_cb
            result = self.fn(*self.args, **kwargs)
            self.success.emit(result)
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            self.failure.emit(f"{e}\n{tb}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = Logger()
        self.config = Config()
        
        # Initialize core components
        self.hw_check = HardwareCheck()
        self.model_manager = ModelManager()
        self.data_loader = DataLoader()
        self.ollama_bridge = OllamaBridge()
        
        self.current_model = None
        self.current_tokenizer = None
        self.current_dataset = None
        self.trainer = None
        self.inference_engine = None
        self.active_worker = None
        self.translator = QTranslator()
        self.saved_geometry = self.config.get("window_geometry", None)
        
        self.init_ui()
        self.load_config()
        self.run_integrity_check()
        self.start_ram_watchdog()
        self.apply_dependency_states()
        if hasattr(QtCore, "qRegisterMetaType"):
            QtCore.qRegisterMetaType(QTextCursor)
        self.online_download_thread = None
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AI Model Trainer Tool")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setCurrentIndex(0)
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_wizard_tab()
        self.create_model_data_tab()
        self.create_training_tab()
        self.create_testing_export_tab()
        self.create_ollama_tab()

        # Resize based on content or saved geometry
        self.apply_geometry_preferences()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Apply styles
        self.apply_styles()
        
    def create_wizard_tab(self):
        """Create beginner wizard tab."""
        container = QWidget()
        layout = QVBoxLayout(container)

        self.wizard_step_label = QLabel("Step 1/5: Select model")
        layout.addWidget(self.wizard_step_label)

        self.wizard_content = QTextEdit()
        self.wizard_content.setReadOnly(True)
        self.wizard_content.setMinimumHeight(120)
        layout.addWidget(self.wizard_content)

        btn_layout = QHBoxLayout()
        self.wizard_back = QPushButton("Back")
        self.wizard_next = QPushButton("Next")
        btn_layout.addWidget(self.wizard_back)
        btn_layout.addWidget(self.wizard_next)
        layout.addLayout(btn_layout)

        self.wizard_stage = 0
        self.update_wizard_text()
        self.wizard_back.clicked.connect(self.wizard_prev)
        self.wizard_next.clicked.connect(self.wizard_next_step)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Wizard")
        
    def create_model_data_tab(self):
        """Create model selection, hardware info, and data input tab."""
        container = QWidget()
        layout = QVBoxLayout(container)

        # Hardware info/recs combined
        hw_group = QGroupBox("Hardware & Recommendations")
        hw_layout = QFormLayout(hw_group)
        for key, value in self.hw_check.system_info.items():
            hw_layout.addRow(key.replace('_', ' ').title() + ":", QLabel(str(value)))
        for key, value in self.hw_check.recommendations.items():
            hw_layout.addRow(f"Recommended {key}:", QLabel(str(value)))

        # Settings (language/theme/accelerator)
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "العربية"])
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        settings_layout.addRow("Language:", self.language_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "High Contrast"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        settings_layout.addRow("Theme:", self.theme_combo)

        self.gpu_checkbox = QCheckBox("Use GPU if available")
        self.gpu_checkbox.setChecked(self.hw_check.system_info.get('gpu_available', False))
        settings_layout.addRow("Accelerator:", self.gpu_checkbox)

        self.ocr_available, self.ocr_msg = deps.require("pytesseract", "pytesseract", optional=True)
        self.pdf_available, self.pdf_msg = deps.require("PyPDF2", "PyPDF2", optional=True)
        self.html_available, self.html_msg = deps.require("bs4", "BeautifulSoup (bs4)", optional=True)

        top_split = QSplitter(Qt.Vertical)
        top_split.addWidget(hw_group)
        top_split.addWidget(settings_group)
        layout.addWidget(top_split)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Popular models
        popular_label = QLabel("Popular Models:")
        model_layout.addWidget(popular_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_manager.get_popular_models())
        self.model_combo.setEditable(True)
        model_layout.addWidget(self.model_combo)

        rec_layout = QHBoxLayout()
        self.recommended_combo = QComboBox()
        self.recommended_combo.addItems(self.model_manager.get_top_models(limit=10))
        rec_refresh = QPushButton("Refresh recommendations")
        rec_refresh.clicked.connect(self.refresh_recommended_models)
        rec_layout.addWidget(QLabel("Recommended:"))
        rec_layout.addWidget(self.recommended_combo)
        rec_layout.addWidget(rec_refresh)
        model_layout.addLayout(rec_layout)

        self.recommended_combo.currentTextChanged.connect(lambda text: self.model_combo.setCurrentText(text))
        
        # Model info
        self.model_info_label = QLabel("Select a model to see information")
        model_layout.addWidget(self.model_info_label)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        btn_row.addWidget(load_btn)

        unload_btn = QPushButton("Unload Model")
        unload_btn.clicked.connect(self.unload_model)
        btn_row.addWidget(unload_btn)

        open_hf_btn = QPushButton("Open on Hugging Face")
        open_hf_btn.clicked.connect(self.open_hf_page)
        btn_row.addWidget(open_hf_btn)

        model_layout.addLayout(btn_row)

        self.model_load_progress = QProgressBar()
        self.model_load_progress.setVisible(True)
        self.model_load_progress.setMaximum(100)
        self.model_load_progress.setValue(0)
        model_layout.addWidget(self.model_load_progress)
        
        layout.addWidget(model_group)
        
        # Data input group
        data_group = QGroupBox("Data Input")
        data_layout = QVBoxLayout(data_group)
        
        self.data_text = QTextEdit()
        self.data_text.setPlaceholderText("Enter training text here...")
        data_layout.addWidget(self.data_text)
        
        # PDF upload
        pdf_layout = QHBoxLayout()
        self.pdf_path_label = QLabel("No PDF selected")
        pdf_layout.addWidget(self.pdf_path_label)

        pdf_btn = QPushButton("Upload PDF")
        pdf_btn.clicked.connect(self.upload_pdf)
        pdf_layout.addWidget(pdf_btn)

        data_layout.addLayout(pdf_layout)

        # OCR option
        ocr_row = QHBoxLayout()
        self.ocr_checkbox = QCheckBox("Use OCR fallback for PDF")
        self.ocr_checkbox.setChecked(True)
        ocr_row.addWidget(self.ocr_checkbox)
        self.ocr_lang_combo = QComboBox()
        self.ocr_lang_combo.addItems(["Arabic+English", "Arabic", "English"])
        ocr_row.addWidget(QLabel("OCR Language:"))
        ocr_row.addWidget(self.ocr_lang_combo)
        data_layout.addLayout(ocr_row)

        self.auto_clean_checkbox = QCheckBox("Auto-clean data (medium)")
        self.auto_clean_checkbox.setChecked(True)
        self.auto_clean_checkbox.stateChanged.connect(self.toggle_auto_clean)
        data_layout.addWidget(self.auto_clean_checkbox)

        clean_row = QHBoxLayout()
        clean_row.addWidget(QLabel("Cleaning level:"))
        self.clean_level_combo = QComboBox()
        self.clean_level_combo.addItems(["Off", "Medium", "Strong"])
        self.clean_level_combo.currentTextChanged.connect(self.on_clean_level_changed)
        clean_row.addWidget(self.clean_level_combo)
        self.lang_detect_checkbox = QCheckBox("Language detection")
        self.lang_detect_checkbox.setChecked(True)
        self.lang_detect_checkbox.stateChanged.connect(self.toggle_lang_detect)
        clean_row.addWidget(self.lang_detect_checkbox)
        data_layout.addLayout(clean_row)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Extraction backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Auto", "Fast", "Enhanced"])
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        backend_row.addWidget(self.backend_combo)
        data_layout.addLayout(backend_row)

        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("OCR engine:"))
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItems(["Auto", "Tesseract", "EasyOCR"])
        self.ocr_engine_combo.currentTextChanged.connect(self.on_engine_changed)
        engine_row.addWidget(self.ocr_engine_combo)
        data_layout.addLayout(engine_row)

        cache_row = QHBoxLayout()
        self.cache_checkbox = QCheckBox("Enable OCR cache")
        self.cache_checkbox.setChecked(True)
        self.cache_checkbox.stateChanged.connect(self.toggle_cache)
        cache_row.addWidget(self.cache_checkbox)
        cache_row.addWidget(QLabel("Max workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 8)
        self.workers_spin.setValue(0)
        self.workers_spin.setToolTip("0 = auto")
        self.workers_spin.valueChanged.connect(self.on_workers_changed)
        cache_row.addWidget(self.workers_spin)
        data_layout.addLayout(cache_row)

        extras_row = QHBoxLayout()
        self.tables_checkbox = QCheckBox("Extract tables")
        self.tables_checkbox.setChecked(False)
        self.tables_checkbox.stateChanged.connect(self.toggle_tables)
        extras_row.addWidget(self.tables_checkbox)
        self.spell_checkbox = QCheckBox("Spellcheck (latin)")
        self.spell_checkbox.setChecked(False)
        self.spell_checkbox.stateChanged.connect(self.toggle_spellcheck)
        extras_row.addWidget(self.spell_checkbox)
        data_layout.addLayout(extras_row)

        # URL import
        url_layout = QVBoxLayout()
        url_label = QLabel("Import from URL:")
        url_layout.addWidget(url_label)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com/file.pdf or .html/.txt/.png")
        url_layout.addWidget(self.url_input)

        url_type_layout = QHBoxLayout()
        url_type_layout.addWidget(QLabel("Content type:"))
        self.url_type_combo = QComboBox()
        self.url_type_combo.addItems(["auto", "pdf", "html", "text", "image"])
        url_type_layout.addWidget(self.url_type_combo)
        url_layout.addLayout(url_type_layout)

        url_btn = QPushButton("تحميل ومعاينة")
        url_btn.clicked.connect(self.import_from_url)
        self.url_download_btn = url_btn
        url_layout.addWidget(url_btn)

        self.url_progress = QProgressBar()
        self.url_progress.setVisible(False)
        url_layout.addWidget(self.url_progress)

        data_layout.addLayout(url_layout)
        
        # Data preview
        preview_btn = QPushButton("Preview Data")
        preview_btn.clicked.connect(self.preview_data)
        data_layout.addWidget(preview_btn)
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        data_layout.addWidget(self.data_preview)

        layout.addWidget(data_group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Model & Hardware")
        
    def create_training_tab(self):
        """Create training tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Training settings
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simple", "Full", "Partial"])
        settings_layout.addRow("Mode:", self.mode_combo)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(self.hw_check.recommendations.get("batch_size", 4))
        settings_layout.addRow("Batch Size:", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(self.hw_check.recommendations.get("epochs", 3))
        settings_layout.addRow("Epochs:", self.epochs_spin)

        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(32, 1024)
        self.max_length_spin.setValue(self.hw_check.recommendations.get("max_length", 128))
        settings_layout.addRow("Max length:", self.max_length_spin)

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 32)
        self.grad_accum_spin.setValue(1)
        settings_layout.addRow("Grad accumulation:", self.grad_accum_spin)

        self.train_workers_spin = QSpinBox()
        self.train_workers_spin.setRange(0, 8)
        self.train_workers_spin.setValue(0)
        self.train_workers_spin.setToolTip("DataLoader workers (0 = main thread)")
        settings_layout.addRow("Data loader workers:", self.train_workers_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(42)
        settings_layout.addRow("Seed:", self.seed_spin)

        self.advanced_toggle = QCheckBox("Show advanced options")
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.stateChanged.connect(self.toggle_advanced)
        settings_layout.addRow(self.advanced_toggle)
        self.set_advanced_visible(False)
        
        layout.addWidget(settings_group)
        
        # Progress and control
        control_group = QGroupBox("Training Control")
        control_layout = QVBoxLayout(control_group)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("ETA: --")
        control_layout.addWidget(self.eta_label)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(200)
        control_layout.addWidget(self.training_log)
        
        layout.addWidget(control_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Training")
        
    def create_testing_export_tab(self):
        """Create testing and export tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Testing group
        test_group = QGroupBox("Model Testing")
        test_layout = QVBoxLayout(test_group)
        inf_defaults = self.config.get_inference_defaults()
        
        self.test_input = QTextEdit()
        self.test_input.setPlaceholderText("Enter test prompt...")
        self.test_input.setMaximumHeight(60)
        test_layout.addWidget(self.test_input)

        decode_row = QFormLayout()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.5)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(float(inf_defaults.get("temperature", 0.2)))
        decode_row.addRow("Temperature:", self.temp_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(float(inf_defaults.get("top_p", 0.8)))
        decode_row.addRow("Top-p:", self.top_p_spin)

        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(0.8, 2.0)
        self.rep_penalty_spin.setSingleStep(0.05)
        self.rep_penalty_spin.setValue(float(inf_defaults.get("repetition_penalty", 1.1)))
        decode_row.addRow("Repetition penalty:", self.rep_penalty_spin)

        self.max_new_tokens_spin = QSpinBox()
        self.max_new_tokens_spin.setRange(8, 512)
        self.max_new_tokens_spin.setValue(int(inf_defaults.get("max_new_tokens", 64)))
        decode_row.addRow("Max new tokens:", self.max_new_tokens_spin)

        self.infer_seed_spin = QSpinBox()
        self.infer_seed_spin.setRange(0, 1_000_000)
        self.infer_seed_spin.setValue(int(inf_defaults.get("seed", 42)))
        decode_row.addRow("Seed:", self.infer_seed_spin)

        test_layout.addLayout(decode_row)
        
        test_btn = QPushButton("Generate Response")
        test_btn.clicked.connect(self.test_model)
        self.test_btn = test_btn
        self.test_btn.setEnabled(False)
        test_layout.addWidget(test_btn)
        
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        test_layout.addWidget(self.test_output)
        
        layout.addWidget(test_group)
        
        # Export group
        export_group = QGroupBox("Export Model")
        export_layout = QVBoxLayout(export_group)
        self.export_combo = QComboBox()
        self.export_combo.addItems(["PyTorch", "SafeTensors", "ONNX", "GGUF"])
        export_layout.addWidget(QLabel("Export format:"))
        export_layout.addWidget(self.export_combo)
        self.export_size_label = QLabel("Estimated size shown after training.")
        export_layout.addWidget(self.export_size_label)

        export_btn = QPushButton("Export Model")
        export_btn.clicked.connect(self.export_model)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)

        ollama_group = QGroupBox("Export to Ollama")
        ollama_layout = QFormLayout(ollama_group)
        self.ollama_name_input = QLineEdit()
        self.ollama_name_input.setPlaceholderText("my-ollama-model")
        ollama_layout.addRow("Ollama model name:", self.ollama_name_input)
        self.ollama_export_btn = QPushButton("Export GGUF + Create in Ollama")
        self.ollama_export_btn.clicked.connect(self.export_to_ollama)
        ollama_layout.addRow(self.ollama_export_btn)
        layout.addWidget(ollama_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Testing & Export")
        
    def create_ollama_tab(self):
        """Create Ollama integration tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Ollama status
        status_group = QGroupBox("Ollama Status")
        status_layout = QVBoxLayout(status_group)
        
        self.ollama_status_label = QLabel()
        self.update_ollama_status()
        status_layout.addWidget(self.ollama_status_label)
        
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self.update_ollama_status)
        status_layout.addWidget(refresh_btn)
        
        layout.addWidget(status_group)
        
        # Model management
        model_group = QGroupBox("Model Management")
        model_layout = QVBoxLayout(model_group)
        
        self.ollama_models_list = QListWidget()
        self.update_ollama_models()
        model_layout.addWidget(self.ollama_models_list)
        
        run_btn = QPushButton("Run Selected Model")
        run_btn.clicked.connect(self.run_ollama_model)
        model_layout.addWidget(run_btn)
        
        layout.addWidget(model_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Ollama")
        
    def apply_styles(self):
        """Apply custom styles."""
        theme = self.config.get("theme", "light")
        qss_file = f"{theme}.qss"
        if theme == "high_contrast":
            qss_file = "high_contrast.qss"
        qss_path = os.path.join(os.path.dirname(__file__), "styles", qss_file)
        try:
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
        except Exception:
            # fallback minimal style
            self.setStyleSheet("")

    def apply_dependency_states(self):
        """Disable UI elements when dependencies are missing."""
        self.disable_button_if(not self.hw_check.system_info.get('gpu_available'), self.gpu_checkbox, "GPU not detected")
        if hasattr(self, 'pdf_path_label'):
            self.disable_button_if(not self.pdf_available, self.pdf_path_label, self.pdf_msg if hasattr(self, 'pdf_msg') else "PDF deps missing")
        # HTML/OCR controls live in data tab; they will be disabled at action time as well.

    def disable_button_if(self, condition, widget, tooltip):
        if condition:
            if hasattr(widget, "setEnabled"):
                widget.setEnabled(False)
            if hasattr(widget, "setToolTip"):
                widget.setToolTip(tooltip)
        else:
            if hasattr(widget, "setEnabled"):
                widget.setEnabled(True)

    def get_ocr_lang(self):
        choice = self.ocr_lang_combo.currentText().lower()
        if "arabic+english" in choice:
            return "ara+eng"
        if "arabic" in choice:
            return "ara"
        return "eng"

    def toggle_auto_clean(self, state):
        self.data_loader.auto_clean = state == Qt.Checked
        self.config.set("auto_clean", self.data_loader.auto_clean)

    def on_clean_level_changed(self, text):
        level = text.lower()
        mapped = "off" if "off" in level else ("strong" if "strong" in level else "medium")
        self.data_loader.clean_level = mapped
        self.config.set("data_clean_level", mapped)

    def toggle_lang_detect(self, state):
        self.data_loader.use_lang_detect = state == Qt.Checked
        self.config.set("data_lang_detect", self.data_loader.use_lang_detect)

    def on_backend_changed(self, text):
        lower = text.lower()
        if "fast" in lower:
            self.data_loader.extract_backend = "fast"
        elif "enhanced" in lower:
            self.data_loader.extract_backend = "enhanced"
            if not shutil.which("ocrmypdf"):
                QMessageBox.warning(self, "Dependency", "ocrmypdf CLI not found. Install system package (with qpdf/ghostscript).")
        else:
            self.data_loader.extract_backend = "auto"
        self.config.set("extract_backend", self.data_loader.extract_backend)

    def on_engine_changed(self, text):
        lower = text.lower()
        if "easy" in lower:
            self.data_loader.ocr_engine = "easyocr"
        elif "tesseract" in lower:
            self.data_loader.ocr_engine = "tesseract"
        else:
            self.data_loader.ocr_engine = "auto"
        self.config.set("ocr_engine", self.data_loader.ocr_engine)

    def toggle_cache(self, state):
        self.data_loader.use_cache = state == Qt.Checked
        self.config.set("use_cache", self.data_loader.use_cache)

    def on_workers_changed(self, val):
        self.data_loader.max_workers = None if val == 0 else val
        self.config.set("max_workers", val)

    def toggle_tables(self, state):
        self.data_loader.extract_tables = state == Qt.Checked
        self.config.set("extract_tables", self.data_loader.extract_tables)

    def toggle_spellcheck(self, state):
        self.data_loader.post_spellcheck = state == Qt.Checked
        self.config.set("post_spellcheck", self.data_loader.post_spellcheck)

    def load_config(self):
        """Load configuration settings."""
        lang = self.config.get("language", "en")
        theme = self.config.get("theme", "light")
        win_geom = self.config.get("window_geometry", None)
        auto_clean = self.config.get("auto_clean", True)
        backend = self.config.get("extract_backend", "auto")
        use_cache = self.config.get("use_cache", True)
        ocr_engine = self.config.get("ocr_engine", "auto")
        max_workers = self.config.get("max_workers", 0)
        extract_tables = self.config.get("extract_tables", False)
        post_spell = self.config.get("post_spellcheck", False)
        clean_level = self.config.get("data_clean_level", "medium")
        lang_detect = self.config.get("data_lang_detect", True)
        train_defaults = self.config.get_training_defaults()
        infer_defaults = self.config.get_inference_defaults()
        self.language_combo.setCurrentText("العربية" if lang == "ar" else "English")
        if theme == "high_contrast":
            self.theme_combo.setCurrentText("High Contrast")
        else:
            self.theme_combo.setCurrentText("Dark" if theme == "dark" else "Light")
        if lang == "ar":
            self.switch_language("ar")
        self.apply_styles()
        self.auto_clean_checkbox.setChecked(bool(auto_clean))
        backend_map = {"auto": "Auto", "fast": "Fast", "enhanced": "Enhanced"}
        self.backend_combo.setCurrentText(backend_map.get(backend, "Auto"))
        engine_map = {"auto": "Auto", "tesseract": "Tesseract", "easyocr": "EasyOCR"}
        self.ocr_engine_combo.setCurrentText(engine_map.get(ocr_engine, "Auto"))
        level_map = {"off": "Off", "medium": "Medium", "strong": "Strong"}
        self.clean_level_combo.setCurrentText(level_map.get(clean_level, "Medium"))
        self.lang_detect_checkbox.setChecked(bool(lang_detect))
        self.cache_checkbox.setChecked(bool(use_cache))
        try:
            self.workers_spin.setValue(int(max_workers))
        except Exception:
            self.workers_spin.setValue(0)
        self.tables_checkbox.setChecked(bool(extract_tables))
        self.spell_checkbox.setChecked(bool(post_spell))
        try:
            self.batch_size_spin.setValue(int(train_defaults.get("batch_size", self.batch_size_spin.value())))
            self.epochs_spin.setValue(int(train_defaults.get("epochs", self.epochs_spin.value())))
            self.max_length_spin.setValue(int(train_defaults.get("max_length", self.max_length_spin.value())))
            self.grad_accum_spin.setValue(int(train_defaults.get("grad_accum", self.grad_accum_spin.value())))
            self.train_workers_spin.setValue(int(train_defaults.get("dataloader_workers", self.train_workers_spin.value())))
            self.seed_spin.setValue(int(train_defaults.get("seed", self.seed_spin.value())))
        except Exception:
            pass
        try:
            if hasattr(self, "temp_spin"):
                self.temp_spin.setValue(float(infer_defaults.get("temperature", self.temp_spin.value())))
                self.top_p_spin.setValue(float(infer_defaults.get("top_p", self.top_p_spin.value())))
                self.rep_penalty_spin.setValue(float(infer_defaults.get("repetition_penalty", self.rep_penalty_spin.value())))
                self.max_new_tokens_spin.setValue(int(infer_defaults.get("max_new_tokens", self.max_new_tokens_spin.value())))
                self.infer_seed_spin.setValue(int(infer_defaults.get("seed", self.infer_seed_spin.value())))
        except Exception:
            pass
        if win_geom and isinstance(win_geom, list) and len(win_geom) == 4:
            self.setGeometry(*win_geom)
        
    # Event handlers
    def load_model(self):
        """Load selected model."""
        model_name = self.model_combo.currentText()
        if not model_name or len(model_name.strip()) < 2:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
        try:
            model_name.encode("ascii")
        except UnicodeEncodeError:
            QMessageBox.warning(self, "Warning", "Model id must be ASCII (Hugging Face id).")
            return
        if self.gpu_checkbox.isChecked() and not self.hw_check.system_info.get('gpu_available'):
            QMessageBox.warning(self, "Warning", "GPU not detected; switching to CPU.")
            self.gpu_checkbox.setChecked(False)
        ok, msg = deps.require("transformers", "transformers", optional=True)
        ok_t, msg_t = deps.require("torch", "PyTorch", optional=True)
        if not (ok and ok_t):
            QMessageBox.critical(self, "Missing dependency", msg if not ok else msg_t)
            return
            
        self.status_bar.showMessage("Loading model...")
        self.model_combo.setEnabled(False)
        self.model_load_progress.setValue(0)
        self.model_load_progress.setMaximum(100)

        # Get model info
        info = self.model_manager.get_model_info(model_name)
        self.model_info_label.setText(
            f"Size: {info['size_gb']}GB | Downloads: {info['downloads']} | License: {info['license']}"
        )
        
        # License confirmation
        license_id = self.model_manager.fetch_license(model_name)
        if license_id not in ("apache-2.0", "mit", "bsd-3-clause", "lgpl-3.0"):
            consent = QMessageBox.question(
                self,
                "License Warning",
                f"Model license is '{license_id}'. Commercial use may be restricted. Continue?",
            )
            if consent != QMessageBox.Yes:
                self.status_bar.showMessage("Load cancelled due to license")
                self.model_combo.setEnabled(True)
                return

        # Check compatibility
        compatible, msg = self.hw_check.check_model_compatibility(info['size_gb'])
        if not compatible:
            QMessageBox.warning(self, "Compatibility Warning", msg)

        def progress(msg):
            self.status_bar.showMessage(msg)
            self.training_log.append(msg) if hasattr(self, 'training_log') else None

        def load_job(progress_value=None):
            success, message = self.model_manager.load_model(
                model_name,
                progress_callback=progress,
                progress_percent=progress_value,
                device_preference='cuda' if self.gpu_checkbox.isChecked() else 'cpu',
            )
            return success, message

        self.active_worker = Worker(load_job, expects_value=True)
        self.active_worker.progress.connect(lambda m: self.status_bar.showMessage(m))
        self.active_worker.progress_value.connect(self._on_model_load_progress)
        self.active_worker.success.connect(lambda res: self._on_model_loaded(res))
        self.active_worker.failure.connect(lambda err: self._on_model_failed(err))
        self.active_worker.start()

    def _on_model_loaded(self, result):
        success, msg = result
        if success:
            self.current_model = self.model_manager.loaded_model
            self.current_tokenizer = self.model_manager.loaded_tokenizer
            self.status_bar.showMessage("Model loaded successfully")
            self.test_btn.setEnabled(True)
            QMessageBox.information(self, "Success", msg)
        else:
            self.status_bar.showMessage("Model loading failed")
            QMessageBox.critical(self, "Error", msg)
        self.model_combo.setEnabled(True)
        self.model_load_progress.setValue(100 if success else 0)

    def _on_model_failed(self, err):
        self.status_bar.showMessage("Model loading failed")
        QMessageBox.critical(self, "Error", err)
        self.model_combo.setEnabled(True)
        self.model_load_progress.setValue(0)

    def _on_model_load_progress(self, value):
        try:
            val = int(value)
        except Exception:
            val = 0
        self.model_load_progress.setValue(max(0, min(val, 100)))

    def unload_model(self):
        self.model_manager.unload_model()
        self.current_model = None
        self.current_tokenizer = None
        gc.collect()
        self.status_bar.showMessage("Model unloaded")

    def open_hf_page(self):
        model_id = self.model_combo.currentText().strip()
        if model_id:
            url = f"https://huggingface.co/{model_id}"
        else:
            url = "https://huggingface.co/models"
        webbrowser.open(url)
        
    def upload_pdf(self):
        """Upload PDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_path_label.setText(os.path.basename(file_path))
            try:
                ok, msg = deps.require("PyPDF2", "PyPDF2/pdfplumber", optional=True)
                if not ok:
                    raise ImportError(msg)
                lang = self.get_ocr_lang()
                self.current_dataset = self.data_loader.load_pdf_data(file_path, use_ocr=self.ocr_checkbox.isChecked(), lang=lang)
                self.status_bar.showMessage("PDF loaded successfully")
                self.show_dataset_preview(self.current_dataset)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")
                
    def preview_data(self):
        """Preview loaded data."""
        if self.current_dataset:
            self.show_dataset_preview(self.current_dataset)
        else:
            # Try to load from text input
            text = self.data_text.toPlainText()
            if text:
                try:
                    dataset = self.data_loader.load_text_data(text)
                    self.current_dataset = dataset
                    self.show_dataset_preview(dataset)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to process text: {str(e)}")
            else:
                QMessageBox.warning(self, "Warning", "No data to preview")

    def show_dataset_preview(self, dataset):
        preview = self.data_loader.preview_data(dataset)
        stats = getattr(self.data_loader, "last_clean_stats", None)
        lines = []
        if stats and stats.get("original"):
            lines.append(f"Cleaned: kept {stats.get('kept',0)}/{stats.get('original',0)} (removed {stats.get('removed',0)})")
        qstats = getattr(self.data_loader, "last_quality_stats", None)
        if qstats and qstats.get("pages") is not None:
            ocr_used = "yes" if qstats.get("ocr_used") else "no"
            cache_hit = "yes" if qstats.get("cache_hit") else "no"
            lines.append(f"OCR enhanced: {ocr_used} | cache: {cache_hit} | pages: {qstats.get('pages',0)} | poor pages: {qstats.get('poor_pages',0)}")
        lines.extend(preview)
        self.data_preview.setText("\n".join(lines))

    def import_from_url(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Warning", "Please enter a URL.")
            return
        ok, msg = deps.require("requests", "requests", optional=True)
        if not ok:
            QMessageBox.critical(self, "Missing dependency", msg)
            return
        content_type = self.url_type_combo.currentText()
        use_ocr = self.ocr_checkbox.isChecked()
        lang = self.get_ocr_lang()
        self.url_progress.setVisible(True)
        self.url_progress.setMaximum(0)
        self.url_download_btn.setEnabled(False)

        def job():
            ds = self.data_loader.load_url(url, content_type=content_type, use_ocr=use_ocr, lang=lang)
            return ds

        self.active_worker = Worker(job)
        self.active_worker.success.connect(self._on_url_loaded)
        self.active_worker.failure.connect(self._on_url_failed)
        self.active_worker.start()

    def _on_url_loaded(self, dataset):
        self.url_progress.setMaximum(1)
        self.url_download_btn.setEnabled(True)
        if dataset:
            self.current_dataset = dataset
            self.show_dataset_preview(dataset)
            self.status_bar.showMessage("URL data loaded")
        else:
            QMessageBox.warning(self, "Warning", "No text extracted from downloaded data")

    def _on_url_failed(self, err):
        self.url_progress.setMaximum(1)
        self.url_download_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Download failed: {err}")
        self.status_bar.showMessage("Download failed")
    def start_training(self):
        """Start model training."""
        if not self.current_model or not self.current_tokenizer:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        # Get data
        if not self.current_dataset:
            text = self.data_text.toPlainText()
            if not text:
                QMessageBox.warning(self, "Warning", "Please provide training data")
                return
            try:
                self.current_dataset = self.data_loader.load_text_data(text)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process data: {str(e)}")
                return
        
        mode = self.mode_combo.currentText().lower()
        batch_size = self.batch_size_spin.value()
        epochs = self.epochs_spin.value()
        max_length = self.max_length_spin.value()
        grad_accum = self.grad_accum_spin.value()
        train_workers = self.train_workers_spin.value()
        seed = self.seed_spin.value()
        
        total_samples = len(self.current_dataset) if self.current_dataset else 0
        confirm = QMessageBox.question(
            self,
            "Confirm Training",
            f"Model: {self.model_combo.currentText()}\n"
            f"Samples: {total_samples}\n"
            f"Batch size: {batch_size}\n"
            f"Epochs: {epochs}\n"
            f"Max length: {max_length}\n"
            f"Grad accumulation: {grad_accum}\n"
            f"Estimated ETA: {self.hw_check.recommendations.get('epochs', epochs)*2} min (approx)\n\nStart training?",
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            self.trainer = ModelTrainer(self.current_model, self.current_tokenizer, self.current_dataset)
            training_args = self.trainer.setup_training(
                mode=mode,
                batch_size=batch_size,
                epochs=epochs,
                max_length=max_length,
                grad_accum=grad_accum,
                dataloader_workers=train_workers,
                seed=seed,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set up training: {e}")
            self.status_bar.showMessage("Training setup failed")
            return
        self.eta_label.setText(f"ETA: ~{self.trainer.last_eta}s")
        
        self.status_bar.showMessage("Training started...")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        def train_job(progress_callback=None):
            return self.trainer.train(progress_callback=progress_callback)

        self.active_worker = Worker(train_job, expects_progress=True)
        self.active_worker.progress_pct.connect(self._on_training_progress)
        self.active_worker.success.connect(self._on_training_finished)
        self.active_worker.failure.connect(self._on_training_failed)
        self.active_worker.start()

    def stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.status_bar.showMessage("Stopping training...")

    def _on_training_finished(self, result):
        success, msg = result
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(success)
        if success:
            self.status_bar.showMessage("Training completed")
            QMessageBox.information(self, "Success", msg)
        else:
            self.status_bar.showMessage("Training failed")
            QMessageBox.critical(self, "Error", msg)

    def _on_training_failed(self, err):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.status_bar.showMessage("Training failed")
        QMessageBox.critical(self, "Error", err)

    def _on_training_progress(self, pct, eta):
        pct = max(0.0, min(pct, 1.0)) if pct is not None else 0
        self.progress_bar.setValue(int(pct * 100))
        if eta is not None and eta >= 0:
            self.eta_label.setText(f"ETA: ~{eta:.1f}s")
        
    def test_model(self):
        """Test the trained model."""
        if not self.current_model or not self.current_tokenizer:
            QMessageBox.warning(self, "Warning", "Please load and train a model first")
            return
            
        prompt = self.test_input.toPlainText()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a test prompt")
            return
            
        self.inference_engine = InferenceEngine(self.current_model, self.current_tokenizer)
        response = self.inference_engine.generate_text(
            prompt,
            max_length=self.max_new_tokens_spin.value(),
            temperature=self.temp_spin.value(),
            top_p=self.top_p_spin.value(),
            repetition_penalty=self.rep_penalty_spin.value(),
            seed=self.infer_seed_spin.value(),
        )
        self.test_output.setText(response)
        
    def export_model(self):
        """Export the trained model."""
        if not self.current_model or not self.current_tokenizer:
            QMessageBox.warning(self, "Warning", "Please load and train a model first")
            return
        ok_tf, msg_tf = deps.require("transformers", "transformers", optional=True)
        ok_t, msg_t = deps.require("torch", "PyTorch", optional=True)
        if not (ok_tf and ok_t):
            QMessageBox.critical(self, "Missing dependency", msg_tf if not ok_tf else msg_t)
            return
            
        # Select export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
            
        selected_format = self.export_combo.currentText().lower()
        formats = [selected_format]

        if not self.hw_check.enough_disk(1):
            QMessageBox.critical(self, "Error", "Not enough disk space to export (need at least 1GB free)")
            return
            
        exporter = ModelExporter(self.current_model, self.current_tokenizer, "")

        self.status_bar.showMessage("Exporting model...")
        self.progress_bar.setMaximum(0)

        def export_job():
            return exporter.export_all(export_dir, formats)

        self.active_worker = Worker(export_job)
        self.active_worker.success.connect(lambda results: self._on_export_finished(results, formats))
        self.active_worker.failure.connect(lambda err: self._on_export_failed(err))
        self.active_worker.start()

    def _on_export_finished(self, results, formats):
        self.progress_bar.setMaximum(1)
        success_count = sum(1 for r in results.values() if r['success'])
        self.status_bar.showMessage("Export finished")
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {success_count}/{len(formats)} formats successfully"
        )

    def _on_export_failed(self, err):
        self.progress_bar.setMaximum(1)
        self.status_bar.showMessage("Export failed")
        QMessageBox.critical(self, "Error", err)

    def export_to_ollama(self):
        """One-click GGUF export and ollama create."""
        if not self.current_model or not self.current_tokenizer:
            QMessageBox.warning(self, "Warning", "Please load and train a model first")
            return
        model_name = self.ollama_name_input.text().strip() or "ai-trainer-model"
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory for Ollama")
        if not export_dir:
            return
        if not self.hw_check.enough_disk(1):
            QMessageBox.critical(self, "Error", "Not enough disk space to export (need at least 1GB free)")
            return

        exporter = ModelExporter(self.current_model, self.current_tokenizer, "")
        target_dir = os.path.join(export_dir, "ollama_export")

        self.status_bar.showMessage("Exporting GGUF and creating Ollama model...")
        self.progress_bar.setMaximum(0)

        def job():
            os.makedirs(target_dir, exist_ok=True)
            ok_gguf, msg_gguf = exporter.export_gguf(target_dir)
            if not ok_gguf:
                return False, msg_gguf
            gguf_path = os.path.join(target_dir, "model.gguf")
            modelfile_path = os.path.join(target_dir, "Modelfile")
            ok_modelfile, msg_modelfile = self.ollama_bridge.create_modelfile(gguf_path, model_name, modelfile_path=modelfile_path)
            if not ok_modelfile:
                return False, f"{msg_gguf}\n{msg_modelfile}"
            create_ok, create_msg = self.ollama_bridge.create_and_register(model_name, modelfile_path=modelfile_path)
            combined_msg = "\n".join([msg_gguf, msg_modelfile, create_msg])
            return create_ok, combined_msg

        self.active_worker = Worker(job)
        self.active_worker.success.connect(lambda res: self._on_ollama_export_finished(res, model_name))
        self.active_worker.failure.connect(lambda err: self._on_export_failed(err))
        self.active_worker.start()

    def _on_ollama_export_finished(self, result, model_name):
        success, msg = result
        self.progress_bar.setMaximum(1)
        self.status_bar.showMessage("Ollama export finished" if success else "Ollama export failed")
        if success:
            QMessageBox.information(self, "Ollama", msg)
            self.update_ollama_models()
        else:
            QMessageBox.critical(self, "Ollama", msg)
        
    def update_ollama_status(self):
        """Update Ollama status display."""
        if self.ollama_bridge.ollama_available:
            self.ollama_status_label.setText("Ollama is installed and available")
        else:
            instructions = self.ollama_bridge.get_install_instructions()
            self.ollama_status_label.setText(f"Ollama not found. {instructions}")
            
    def update_ollama_models(self):
        """Update list of Ollama models."""
        self.ollama_models_list.clear()
        models = self.ollama_bridge.list_ollama_models()
        self.ollama_models_list.addItems(models)
        
    def run_ollama_model(self):
        """Run selected Ollama model."""
        selected = self.ollama_models_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
            
        model_name = selected.text()
        success, msg = self.ollama_bridge.run_ollama_model(model_name)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Error", msg)

    # Wizard helpers
    def update_wizard_text(self):
        steps = [
            "Choose or search a small model (distilgpt2 recommended).",
            "Enter training text or upload a PDF.",
            "Select mode (Simple) and start training.",
            "Export to PyTorch/SafeTensors/ONNX/GGUF.",
            "If Ollama is installed, create and run the model.",
        ]
        self.wizard_step_label.setText(f"Step {self.wizard_stage+1}/5")
        self.wizard_content.setText(steps[self.wizard_stage])
        self.wizard_back.setEnabled(self.wizard_stage > 0)
        if self.wizard_stage == len(steps) - 1:
            self.wizard_next.setText("Finish")
        else:
            self.wizard_next.setText("Next")

    def wizard_prev(self):
        if self.wizard_stage > 0:
            self.wizard_stage -= 1
            self.update_wizard_text()

    def wizard_next_step(self):
        if self.wizard_stage < 4:
            self.wizard_stage += 1
            self.update_wizard_text()

    def run_integrity_check(self):
        missing = integrity.missing(base_dir=".")
        warn_lines = []
        if missing:
            warn_lines.append("Missing modules:\n" + "\n".join(missing))
        warn_lines.extend(integrity.env_warnings(base_dir="."))
        if warn_lines:
            QMessageBox.warning(
                self,
                "Environment warnings",
                "\n\n".join(warn_lines),
            )

    def on_language_changed(self, text):
        lang = "ar" if "الع" in text else "en"
        self.switch_language(lang)
        self.config.set("language", lang)
        self.apply_dependency_states()

    def switch_language(self, lang_code):
        if lang_code == "ar":
            locale = QLocale(QLocale.Arabic)
            QApplication.instance().setLayoutDirection(Qt.RightToLeft)
        else:
            locale = QLocale(QLocale.English)
            QApplication.instance().setLayoutDirection(Qt.LeftToRight)
        QLocale.setDefault(locale)

    def on_theme_changed(self, text):
        lower = text.lower()
        if "contrast" in lower:
            theme = "high_contrast"
        elif lower == "dark":
            theme = "dark"
        else:
            theme = "light"
        self.config.set("theme", theme)
        self.apply_styles()
        self.apply_dependency_states()

    def refresh_recommended_models(self):
        models = self.model_manager.get_top_models(limit=10)
        self.recommended_combo.clear()
        self.recommended_combo.addItems(models)

    def start_ram_watchdog(self):
        self.ram_timer = QTimer(self)
        self.ram_timer.setInterval(4000)
        self.ram_timer.timeout.connect(self._check_ram)
        self.ram_timer.start()

    def _check_ram(self):
        if memory.near_capacity():
            summary = memory.ram_summary()
            self.status_bar.showMessage(f"RAM high usage: {summary['percent']}%")

    def closeEvent(self, event):
        try:
            self.config.set("language", "ar" if QApplication.instance().layoutDirection() == Qt.RightToLeft else "en")
            self.config.set("theme", self.theme_combo.currentText().lower().replace(" ", "_"))
            self.config.set("auto_clean", self.auto_clean_checkbox.isChecked())
            self.config.set("extract_backend", self.data_loader.extract_backend)
            self.config.set("use_cache", self.cache_checkbox.isChecked())
            self.config.set("ocr_engine", self.data_loader.ocr_engine)
            self.config.set("max_workers", self.workers_spin.value())
            self.config.set("extract_tables", self.tables_checkbox.isChecked())
            self.config.set("post_spellcheck", self.spell_checkbox.isChecked())
            self.config.update_training_defaults(
                batch_size=self.batch_size_spin.value(),
                max_length=self.max_length_spin.value(),
                epochs=self.epochs_spin.value(),
                grad_accum=self.grad_accum_spin.value(),
                dataloader_workers=self.train_workers_spin.value(),
                seed=self.seed_spin.value(),
            )
            if hasattr(self, "temp_spin"):
                self.config.update_inference_defaults(
                    temperature=self.temp_spin.value(),
                    top_p=self.top_p_spin.value(),
                    repetition_penalty=self.rep_penalty_spin.value(),
                    max_new_tokens=self.max_new_tokens_spin.value(),
                    seed=self.infer_seed_spin.value(),
                )
            geom = self.geometry()
            self.config.set("window_geometry", [geom.x(), geom.y(), geom.width(), geom.height()])
        except Exception:
            pass
        super().closeEvent(event)

    def apply_geometry_preferences(self):
        screen = QApplication.primaryScreen().availableGeometry()

        # Force a sensible minimum equal to available screen so controls stay visible when maximized
        self.setMinimumSize(screen.width(), screen.height())

        # If saved geometry exists, honor it but keep min size
        if self.saved_geometry and isinstance(self.saved_geometry, list) and len(self.saved_geometry) == 4:
            self.setGeometry(*self.saved_geometry)
            return

        # Default: fill most of the screen while allowing user resize
        preferred_w, preferred_h = self.compute_preferred_size()
        w = max(preferred_w, screen.width())
        h = max(preferred_h, screen.height())
        self.resize(w, h)

    def compute_preferred_size(self):
        fm = self.fontMetrics()
        tab_texts = ["Wizard", "Model & Hardware", "Training", "Testing & Export", "Ollama"]
        tabs_width = sum(fm.boundingRect(t).width() + 40 for t in tab_texts) + 60
        longest_label = max([
            "Recommended max_length:", "Recommended device:", "Accelerator:", "Import from URL:", "Recommended:"
        ], key=len)
        label_width = fm.boundingRect(longest_label).width() + 220
        text_area_height = fm.height() * 18  # room for preview and controls
        min_width = max(1200, tabs_width, label_width)
        min_height = max(800, text_area_height)
        return min_width, min_height

    def toggle_advanced(self, state):
        visible = state == Qt.Checked
        self.set_advanced_visible(visible)

    def set_advanced_visible(self, visible):
        for widget in [self.batch_size_spin, self.epochs_spin, self.max_length_spin, self.grad_accum_spin, self.train_workers_spin, self.seed_spin]:
            widget.setVisible(visible)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
