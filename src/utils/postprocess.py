from utils.deps import require

spell_avail, spell_msg = require("pyspellchecker", "pyspellchecker", optional=True)

if spell_avail:
    from spellchecker import SpellChecker  # type: ignore


class SpellPostProcessor:
    def __init__(self, languages=None):
        self.languages = languages or ["en"]
        self.spell = SpellChecker(language=None) if spell_avail else None

    def correct_line(self, text: str, arabic_ratio: float = 0.0):
        if not self.spell or arabic_ratio > 0.3:
            return text
        words = text.split()
        corrected = []
        for w in words:
            if w.isalpha() and len(w) > 3:
                corrected.append(self.spell.correction(w))
            else:
                corrected.append(w)
        return " ".join(corrected)
