from autopress.app_tk import window_title_matches


def test_empty_window_title_filter_allows_any_foreground_window():
    assert window_title_matches("", "Not Defteri")
    assert window_title_matches("   ", "Oyun")


def test_nonempty_window_title_filter_stays_case_insensitive():
    assert window_title_matches("not defteri", "Belge - Not Defteri")
    assert not window_title_matches("oyun", "Not Defteri")
