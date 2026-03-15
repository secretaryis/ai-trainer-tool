import json
import os

class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        # Default config
        return {
            "language": "en",
            "theme": "light",
            "last_model": "",
            "training_defaults": {
                "batch_size": 4,
                "max_length": 128,
                "epochs": 3
            },
            "export_formats": ["pytorch", "safetensors"]
        }

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value
        self.save_config()

    def update_training_defaults(self, batch_size=None, max_length=None, epochs=None):
        """Update training default settings."""
        if batch_size is not None:
            self.config["training_defaults"]["batch_size"] = batch_size
        if max_length is not None:
            self.config["training_defaults"]["max_length"] = max_length
        if epochs is not None:
            self.config["training_defaults"]["epochs"] = epochs
        self.save_config()

    def get_training_defaults(self):
        """Get training default settings."""
        return self.config.get("training_defaults", {})