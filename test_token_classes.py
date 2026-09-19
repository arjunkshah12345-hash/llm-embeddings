from token_classes import assign_classes


def test_assign_classes_separates_surface_form_from_frequency():
    texts = [" ", ".", "the", "zygote", "middle"]
    counts = [100, 50, 80, 1, 5]
    assert assign_classes(texts, counts, common_k=1, rare_max_count=1) == [
        "whitespace",
        "punctuation",
        "common",
        "rare",
        "other",
    ]
