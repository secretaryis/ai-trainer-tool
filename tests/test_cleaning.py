import pytest
from core.data_loader import DataLoader


langdetect_installed = True
try:
    from langdetect import detect  # noqa: F401
except Exception:
    langdetect_installed = False

arabic_shape_installed = True
try:
    import arabic_reshaper  # noqa: F401
    from bidi.algorithm import get_display  # noqa: F401
except Exception:
    arabic_shape_installed = False


@pytest.mark.skipif(not langdetect_installed, reason="langdetect not installed")
def test_mixed_language_not_reordered():
    dl = DataLoader()
    keep, cleaned = dl._clean_line("Hello مرحبا world", level="medium", lang_detection=True)
    assert keep
    assert cleaned.startswith("Hello")
    assert "world" in cleaned


@pytest.mark.skipif(not arabic_shape_installed, reason="arabic shaping libs not installed")
def test_arabic_line_cleaned_and_shaped():
    dl = DataLoader()
    noisy = "|ينفلا ميلعتلاو ميلعتلاو ةيبرتلا ريزو"
    keep, cleaned = dl._clean_line(noisy, level="medium", lang_detection=True)
    assert keep
    assert "|" not in cleaned
    assert len(cleaned) >= len(noisy.replace("|", "").strip())


def test_clean_dataset_filters_noise_and_duplicates():
    dl = DataLoader()
    text = "Hello world\nHello world\nhttp://example.com\n\uFFFD\uFFFD\uFFFD\nok line"
    ds = dl.load_text_data(text, auto_clean=False)
    cleaned = dl.clean_dataset(ds, level="medium", lang_detection=False)
    assert len(cleaned) == 2  # "Hello world" deduped + "ok line"
    assert all("http" not in item["text"] for item in cleaned)
