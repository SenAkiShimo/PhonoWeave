from phonoweave.gui import _calibration_page_html, _main_page_html


def test_main_page_exposes_live_calibration() -> None:
    html = _main_page_html()
    assert 'href="/calibration"' in html
    assert "Live Calibration / 现场校准" in html


def test_calibration_page_exposes_session_restore_and_audio_tools() -> None:
    html = _calibration_page_html()
    assert 'id="open-session"' in html
    assert 'id="audio"' in html
    assert 'id="live"' in html
    assert 'id="take"' in html
