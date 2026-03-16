from core.data_loader import DataLoader


def test_text_split():
    dl = DataLoader()
    ds = dl.load_text_data("hello\nworld")
    assert len(ds) == 2
    preview = dl.preview_data(ds)
    assert preview[0] == "hello"


def test_cleaning_off_keeps_all_lines():
    dl = DataLoader()
    dl.auto_clean = True
    dl.clean_level = "off"
    ds = dl.load_text_data("a\nb\nc")
    assert len(ds) == 3


def test_lang_detect_toggle():
    dl = DataLoader()
    dl.use_lang_detect = False
    ds, stats = dl.clean_dataset(dl.load_text_data("hi\nمرحبا", auto_clean=False), level="medium", lang_detection=dl.use_lang_detect, return_stats=True)
    # Should keep both lines when lang detect is off
    assert stats["kept"] == len(ds)
