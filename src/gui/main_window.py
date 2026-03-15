import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox,
    QProgressBar, QStatusBar, QMessageBox, QFileDialog, QGroupBox,
    QFormLayout, QSpinBox, QCheckBox, QListWidget, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTranslator, QLocale
from PyQt5.QtGui import QFont
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

class Worker(QThread):
    progress = pyqtSignal(str)
    success = pyqtSignal(object)
    failure = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.success.emit(result)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
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
        
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AI Model Trainer Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_wizard_tab()
        self.create_hardware_tab()
        self.create_model_data_tab()
        self.create_training_tab()
        self.create_testing_export_tab()
        self.create_ollama_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Apply styles
        self.apply_styles()
        
    def create_hardware_tab(self):
        """Create hardware and settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Hardware info group
        hw_group = QGroupBox("Hardware Information")
        hw_layout = QFormLayout(hw_group)
        
        self.hw_info_labels = {}
        for key, value in self.hw_check.system_info.items():
            label = QLabel(str(value))
            self.hw_info_labels[key] = label
            hw_layout.addRow(key.replace('_', ' ').title() + ":", label)
        
        layout.addWidget(hw_group)
        
        # Recommendations group
        rec_group = QGroupBox("Recommendations")
        rec_layout = QFormLayout(rec_group)
        
        for key, value in self.hw_check.recommendations.items():
            label = QLabel(str(value))
            rec_layout.addRow(key.replace('_', ' ').title() + ":", label)
        
        layout.addWidget(rec_group)

        ram_label = QLabel(self.hw_check.ram_summary)
        layout.addWidget(ram_label)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "العربية"])
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        settings_layout.addRow("Language:", self.language_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        settings_layout.addRow("Theme:", self.theme_combo)

        self.gpu_checkbox = QCheckBox("Use GPU if available")
        self.gpu_checkbox.setChecked(self.hw_check.system_info.get('gpu_available', False))
        settings_layout.addRow("Accelerator:", self.gpu_checkbox)

        layout.addWidget(settings_group)
        
        self.tab_widget.addTab(tab, "Hardware & Settings")

    def create_wizard_tab(self):
        """Create beginner wizard tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        self.tab_widget.addTab(tab, "Wizard")
        
    def create_model_data_tab(self):
        """Create model selection and data input tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
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
        
        # Model info
        self.model_info_label = QLabel("Select a model to see information")
        model_layout.addWidget(self.model_info_label)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)

        unload_btn = QPushButton("Unload Model")
        unload_btn.clicked.connect(self.unload_model)
        model_layout.addWidget(unload_btn)
        
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
        
        # Data preview
        preview_btn = QPushButton("Preview Data")
        preview_btn.clicked.connect(self.preview_data)
        data_layout.addWidget(preview_btn)
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMaximumHeight(100)
        data_layout.addWidget(self.data_preview)
        
        layout.addWidget(data_group)
        
        self.tab_widget.addTab(tab, "Model & Data")
        
    def create_training_tab(self):
        """Create training tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Training settings
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simple", "Full", "Partial"])
        settings_layout.addRow("Mode:", self.mode_combo)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        settings_layout.addRow("Batch Size:", self.batch_size_spin)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(3)
        settings_layout.addRow("Epochs:", self.epochs_spin)
        
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
        
        self.tab_widget.addTab(tab, "Training")
        
    def create_testing_export_tab(self):
        """Create testing and export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Testing group
        test_group = QGroupBox("Model Testing")
        test_layout = QVBoxLayout(test_group)
        
        self.test_input = QTextEdit()
        self.test_input.setPlaceholderText("Enter test prompt...")
        self.test_input.setMaximumHeight(60)
        test_layout.addWidget(self.test_input)
        
        test_btn = QPushButton("Generate Response")
        test_btn.clicked.connect(self.test_model)
        test_layout.addWidget(test_btn)
        
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        test_layout.addWidget(self.test_output)
        
        layout.addWidget(test_group)
        
        # Export group
        export_group = QGroupBox("Export Model")
        export_layout = QVBoxLayout(export_group)
        
        self.export_formats = ["PyTorch", "SafeTensors", "ONNX", "GGUF"]
        self.export_checks = {}
        for fmt in self.export_formats:
            check = QCheckBox(fmt)
            if fmt in ["PyTorch", "SafeTensors"]:
                check.setChecked(True)
            self.export_checks[fmt] = check
            export_layout.addWidget(check)
        
        export_btn = QPushButton("Export Model")
        export_btn.clicked.connect(self.export_model)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)
        
        self.tab_widget.addTab(tab, "Testing & Export")
        
    def create_ollama_tab(self):
        """Create Ollama integration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
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
        
        self.tab_widget.addTab(tab, "Ollama")
        
    def apply_styles(self):
        """Apply custom styles."""
        theme = self.config.get("theme", "light")
        qss_path = os.path.join(os.path.dirname(__file__), "styles", f"{theme}.qss")
        try:
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())
        except Exception:
            # fallback minimal style
            self.setStyleSheet("")

    def load_config(self):
        """Load configuration settings."""
        lang = self.config.get("language", "en")
        theme = self.config.get("theme", "light")
        self.language_combo.setCurrentText("العربية" if lang == "ar" else "English")
        self.theme_combo.setCurrentText("Dark" if theme == "dark" else "Light")
        if lang == "ar":
            self.switch_language("ar")
        self.apply_styles()
        
    # Event handlers
    def load_model(self):
        """Load selected model."""
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
            
        self.status_bar.showMessage("Loading model...")
        self.model_combo.setEnabled(False)
        
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

        def load_job():
            success, message = self.model_manager.load_model(
                model_name,
                progress_callback=progress,
                device_preference='cuda' if self.gpu_checkbox.isChecked() else 'cpu',
            )
            return success, message

        self.active_worker = Worker(load_job)
        self.active_worker.progress.connect(lambda m: self.status_bar.showMessage(m))
        self.active_worker.success.connect(lambda res: self._on_model_loaded(res))
        self.active_worker.failure.connect(lambda err: self._on_model_failed(err))
        self.active_worker.start()

    def _on_model_loaded(self, result):
        success, msg = result
        if success:
            self.current_model = self.model_manager.loaded_model
            self.current_tokenizer = self.model_manager.loaded_tokenizer
            self.status_bar.showMessage("Model loaded successfully")
            QMessageBox.information(self, "Success", msg)
        else:
            self.status_bar.showMessage("Model loading failed")
            QMessageBox.critical(self, "Error", msg)
        self.model_combo.setEnabled(True)

    def _on_model_failed(self, err):
        self.status_bar.showMessage("Model loading failed")
        QMessageBox.critical(self, "Error", err)
        self.model_combo.setEnabled(True)

    def unload_model(self):
        self.model_manager.unload_model()
        self.current_model = None
        self.current_tokenizer = None
        gc.collect()
        self.status_bar.showMessage("Model unloaded")
        
    def upload_pdf(self):
        """Upload PDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_path_label.setText(os.path.basename(file_path))
            try:
                self.current_dataset = self.data_loader.load_pdf_data(file_path)
                self.status_bar.showMessage("PDF loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")
                
    def preview_data(self):
        """Preview loaded data."""
        if self.current_dataset:
            preview = self.data_loader.preview_data(self.current_dataset)
            self.data_preview.setText("\n".join(preview))
        else:
            # Try to load from text input
            text = self.data_text.toPlainText()
            if text:
                try:
                    dataset = self.data_loader.load_text_data(text)
                    preview = self.data_loader.preview_data(dataset)
                    self.data_preview.setText("\n".join(preview))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to process text: {str(e)}")
            else:
                QMessageBox.warning(self, "Warning", "No data to preview")
                
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
        
        self.trainer = ModelTrainer(self.current_model, self.current_tokenizer, self.current_dataset)
        training_args = self.trainer.setup_training(mode=mode, batch_size=batch_size, epochs=epochs)
        self.eta_label.setText(f"ETA: ~{self.trainer.last_eta}s")
        
        self.status_bar.showMessage("Training started...")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setMaximum(0)

        def train_job():
            return self.trainer.train()

        self.active_worker = Worker(train_job)
        self.active_worker.success.connect(self._on_training_finished)
        self.active_worker.failure.connect(self._on_training_failed)
        self.active_worker.start()

    def stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.status_bar.showMessage("Stopping training...")

    def _on_training_finished(self, result):
        success, msg = result
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(1)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        if success:
            self.status_bar.showMessage("Training completed")
            QMessageBox.information(self, "Success", msg)
        else:
            self.status_bar.showMessage("Training failed")
            QMessageBox.critical(self, "Error", msg)

    def _on_training_failed(self, err):
        self.progress_bar.setMaximum(1)
        self.stop_btn.setEnabled(False)
        self.train_btn.setEnabled(True)
        self.status_bar.showMessage("Training failed")
        QMessageBox.critical(self, "Error", err)
        
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
        response = self.inference_engine.generate_text(prompt)
        self.test_output.setText(response)
        
    def export_model(self):
        """Export the trained model."""
        if not self.current_model or not self.current_tokenizer:
            QMessageBox.warning(self, "Warning", "Please load and train a model first")
            return
            
        # Select export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
            
        formats = [fmt for fmt, check in self.export_checks.items() if check.isChecked()]
        if not formats:
            QMessageBox.warning(self, "Warning", "Please select at least one export format")
            return

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

    def on_language_changed(self, text):
        lang = "ar" if "الع" in text else "en"
        self.switch_language(lang)
        self.config.set("language", lang)

    def switch_language(self, lang_code):
        if lang_code == "ar":
            locale = QLocale(QLocale.Arabic)
            QApplication.instance().setLayoutDirection(Qt.RightToLeft)
        else:
            locale = QLocale(QLocale.English)
            QApplication.instance().setLayoutDirection(Qt.LeftToRight)
        QLocale.setDefault(locale)

    def on_theme_changed(self, text):
        theme = "dark" if text.lower() == "dark" else "light"
        self.config.set("theme", theme)
        self.apply_styles()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
