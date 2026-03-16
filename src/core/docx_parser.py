try:
    import docx
except ImportError:
    docx = None


class DocxParser:
    def extract_text(self, path):
        if docx is None:
            raise ImportError("python-docx not installed. Install to read DOCX files.")
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
