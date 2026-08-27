from phonoweave.gui_page import HTML


def test_gui_page_has_language_switch_and_persistence() -> None:
    assert 'id="lang-zh"' in HTML
    assert 'id="lang-en"' in HTML
    assert "phonoweave.language" in HTML
    assert "navigator.language" in HTML


def test_gui_page_keeps_raw_scientific_terms_visible() -> None:
    assert "split_recommended" in HTML
    assert "strongly_supported" in HTML
    assert "supported_under_proxy" in HTML
    assert "建议区分录制" in HTML
    assert "当前代理指标支持" in HTML
