import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_file="app.log"):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        """Setup logging configuration."""
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def info(self, message):
        """Log info message."""
        logging.info(message)

    def warning(self, message):
        """Log warning message."""
        logging.warning(message)

    def error(self, message):
        """Log error message."""
        logging.error(message)

    def get_logs(self, lines=50):
        """Get last n lines of logs."""
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, 'r') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines