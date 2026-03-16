from core.data_loader import DataLoader


def test_deduplicate():
    dl = DataLoader()
    ds = dl.load_text_data("hi\nhi\nthere")
    filtered = dl.filter_deduplicate(ds)
    assert len(filtered) == 2


def test_min_length():
    dl = DataLoader()
    ds = dl.load_text_data("a b\nshort\nthis is long enough")
    filtered = dl.filter_min_length(ds, min_words=3)
    assert len(filtered) == 1
