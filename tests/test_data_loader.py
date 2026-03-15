from core.data_loader import DataLoader


def test_text_split():
    dl = DataLoader()
    ds = dl.load_text_data("hello\nworld")
    assert len(ds) == 2
    preview = dl.preview_data(ds)
    assert preview[0] == "hello"
