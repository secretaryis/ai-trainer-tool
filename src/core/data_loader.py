from datasets import Dataset
from .pdf_parser import PDFParser
import os

class DataLoader:
    def __init__(self):
        self.pdf_parser = PDFParser()

    def load_text_data(self, text):
        """Load data from plain text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Split text into lines or paragraphs
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        data = {'text': lines}
        return Dataset.from_dict(data)

    def load_pdf_data(self, pdf_path):
        """Load data from PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = self.pdf_parser.extract_text(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")
        
        return self.load_text_data(text)

    def preview_data(self, dataset, max_items=5):
        """Get preview of the dataset."""
        preview = []
        for i, item in enumerate(dataset):
            if i >= max_items:
                break
            preview.append(item['text'][:100] + "..." if len(item['text']) > 100 else item['text'])
        return preview

    def get_dataset_info(self, dataset):
        """Get information about the dataset."""
        return {
            'num_samples': len(dataset),
            'columns': list(dataset.column_names),
        }