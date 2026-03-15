from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import torch
import os
from datetime import datetime
import math

class ModelTrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.trainer = None
        self.should_stop = False
        self.last_eta = None

    def stop(self):
        """Request training stop."""
        self.should_stop = True

    class _StopCallback(TrainerCallback):
        def __init__(self, parent):
            self.parent = parent

        def on_step_begin(self, args, state, control, **kwargs):
            if self.parent.should_stop:
                control.should_training_stop = True
            return control

    def tokenize_function(self, examples):
        """Tokenize the text data."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Will be adjusted based on hardware
        )

    def prepare_dataset(self, max_length=128):
        """Prepare dataset for training."""
        tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            ),
            batched=True,
            remove_columns=["text"]
        )
        return tokenized_dataset

    def setup_training(self, mode='simple', batch_size=4, epochs=3, max_length=128, output_dir=None):
        """Setup training based on mode."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./trained_model_{timestamp}"

        # Prepare dataset
        train_dataset = self.prepare_dataset(max_length)
        total_steps = math.ceil(len(train_dataset) / batch_size) * epochs

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        # Training arguments based on mode
        if mode == 'simple':
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                save_steps=max(50, total_steps // 10),
                save_total_limit=1,
                logging_steps=10,
                learning_rate=5e-5,
                weight_decay=0.01,
                save_strategy="steps",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                report_to=[],
            )
        elif mode == 'full':
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=max(50, total_steps // 10),
                save_total_limit=2,
                logging_steps=10,
                learning_rate=5e-5,
                weight_decay=0.01,
                save_strategy="steps",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                report_to=[],
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

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=max(50, total_steps // 10),
                save_total_limit=2,
                logging_steps=10,
                learning_rate=1e-4,  # Higher LR for fine-tuning
                weight_decay=0.01,
                save_strategy="steps",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                report_to=[],
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

    def train(self, progress_callback=None, resume_from_checkpoint=None):
        """Start training, optionally resuming."""
        if not self.trainer:
            raise ValueError("Training not set up. Call setup_training first.")

        try:
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
        # Rough heuristic: assume ~0.15s per step on CPU small models
        seconds_per_step = 0.15 if not torch.cuda.is_available() else 0.05
        return round(total_steps * seconds_per_step, 1)
