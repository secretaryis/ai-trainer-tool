from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from packaging import version
import transformers
import torch
import os
from datetime import datetime
import math
import random


def _maybe_bf16():
    """Return True only when bf16 is supported to avoid crashes on older GPUs/CPUs."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

class ModelTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.trainer = None
        self.should_stop = False
        self.last_eta = None
        self.tf_version = version.parse(transformers.__version__)
        self.total_steps = None
        self.seconds_per_step = 0.15 if not torch.cuda.is_available() else 0.05
        self.seed = 42
        self.dataloader_workers = 0
        self.grad_accum = 1

    def stop(self):
        """Request training stop."""
        self.should_stop = True

    def _set_seed(self, seed: int):
        self.seed = seed
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Best effort only; keep UI responsive
            pass

    class _StopCallback(TrainerCallback):
        def __init__(self, parent):
            self.parent = parent

        def on_step_begin(self, args, state, control, **kwargs):
            if self.parent.should_stop:
                control.should_training_stop = True
            return control

    class _ProgressCallback(TrainerCallback):
        def __init__(self, parent, progress_cb):
            self.parent = parent
            self.progress_cb = progress_cb

        def on_step_end(self, args, state, control, **kwargs):
            return self._report(state, control)

        def on_log(self, args, state, control, **kwargs):
            return self._report(state, control)

        def _report(self, state, control):
            if not self.progress_cb or not self.parent.total_steps:
                return control
            pct = min(1.0, state.global_step / max(self.parent.total_steps, 1))
            remaining_steps = max(self.parent.total_steps - state.global_step, 0)
            eta = round(remaining_steps * self.parent.seconds_per_step, 1)
            try:
                self.progress_cb(pct, eta)
            except Exception:
                pass
            return control

    def tokenize_function(self, examples):
        """Tokenize the text data."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Will be adjusted based on hardware
        )

    def prepare_dataset(self, max_length=128, num_proc=None):
        """Prepare dataset for training, handling both HF Dataset and list-like fallback."""
        if hasattr(self.dataset, "map"):
            tokenized_dataset = self.dataset.map(
                lambda x: self.tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                ),
                batched=True,
                num_proc=num_proc,
                remove_columns=["text"],
            )
            return tokenized_dataset

        encoded = []
        for row in self.dataset:
            tok = self.tokenizer(
                row["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            encoded.append(tok)
        return encoded

    def setup_training(self, mode='simple', batch_size=4, epochs=3, max_length=128, output_dir=None, grad_accum=1, dataloader_workers=0, seed=42):
        """Setup training based on mode."""
        self._set_seed(seed)
        self.dataloader_workers = dataloader_workers
        self.grad_accum = max(1, grad_accum)
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./trained_model_{timestamp}"

        # Prepare dataset
        max_length = max(8, max_length)
        train_dataset = self.prepare_dataset(max_length, num_proc=None)
        total_steps = math.ceil(len(train_dataset) / max(batch_size, 1)) * max(epochs, 1) if len(train_dataset) else 0
        self.total_steps = total_steps

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        # Training arguments based on mode
        if mode == 'simple':
            training_args = self._build_training_args(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=self.grad_accum,
                save_steps=max(50, total_steps // 10),
                save_total_limit=1,
                logging_steps=10,
                learning_rate=5e-5,
                weight_decay=0.01,
            )
        elif mode == 'full':
            training_args = self._build_training_args(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.grad_accum,
                save_steps=max(50, total_steps // 10),
                save_total_limit=2,
                logging_steps=10,
                learning_rate=5e-5,
                weight_decay=0.01,
            )
        elif mode == 'partial':
            # Freeze all layers except the last few
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze last layer
            if hasattr(self.model, 'lm_head'):
                for param in self.model.lm_head.parameters():
                    param.requires_grad = True
            # Unfreeze last transformer layer
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                for param in self.model.transformer.h[-1].parameters():
                    param.requires_grad = True

            training_args = self._build_training_args(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.grad_accum,
                save_steps=max(50, total_steps // 10),
                save_total_limit=2,
                logging_steps=10,
                learning_rate=1e-4,  # Higher LR for fine-tuning
                weight_decay=0.01,
            )
        else:
            raise ValueError("Invalid training mode")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[self._StopCallback(self)],
        )

        self.last_eta = self._estimate_eta(total_steps, training_args)
        return training_args

    def _build_training_args(self, **kwargs):
        """Construct TrainingArguments compatible with transformers 4.x/5.x."""
        args = dict(kwargs)
        args.setdefault("gradient_accumulation_steps", self.grad_accum)
        args.setdefault("dataloader_num_workers", max(0, self.dataloader_workers or 0))
        args.setdefault("seed", self.seed)

        # Shared defaults
        args.setdefault("save_strategy", "steps")
        args.setdefault("load_best_model_at_end", False)
        args.setdefault("report_to", [])
        if _maybe_bf16():
            args.setdefault("bf16", True)
        elif torch.cuda.is_available():
            args.setdefault("fp16", True)

        # Handle eval strategy naming difference between v4 and v5
        if self.tf_version.major >= 5:
            args["eval_strategy"] = "no"
            args.pop("evaluation_strategy", None)
        else:
            args["evaluation_strategy"] = "no"
            args.pop("eval_strategy", None)

        try:
            return TrainingArguments(**args)
        except TypeError as exc:
            # Fallback: remove any eval/save keys if still incompatible
            safe_args = {k: v for k, v in args.items() if k not in {"evaluation_strategy", "eval_strategy", "save_strategy"}}
            return TrainingArguments(**safe_args)

    def train(self, progress_callback=None, resume_from_checkpoint=None):
        """Start training, optionally resuming."""
        if not self.trainer:
            raise ValueError("Training not set up. Call setup_training first.")

        try:
            if progress_callback:
                self.trainer.add_callback(self._ProgressCallback(self, progress_callback))
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self.trainer.save_model()
            return True, "Training completed successfully."
        except Exception as e:
            return False, f"Training failed: {str(e)}"

    def get_training_logs(self):
        """Get training logs."""
        if self.trainer and hasattr(self.trainer.state, 'log_history'):
            return self.trainer.state.log_history
        return []

    def _estimate_eta(self, total_steps, training_args):
        return round(total_steps * self.seconds_per_step, 1)
