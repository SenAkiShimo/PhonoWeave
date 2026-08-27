from phonoweave.gui_page_remake import HTML


def test_remake_uses_workbench_layout() -> None:
    assert 'class="workbench"' in HTML
    assert 'class="pane inspector-pane"' in HTML
    assert 'class="statusbar"' in HTML
    assert 'PHONOWEAVE' in HTML


def test_remake_does_not_restore_dashboard_cards_or_badges() -> None:
    assert 'class="cards"' not in HTML
    assert 'class="card"' not in HTML
    assert 'class="badge' not in HTML


def test_remake_keeps_bilingual_switch_and_machine_terms() -> None:
    assert '>中文</button>' in HTML
    assert '>EN</button>' in HTML
    assert "localStorage.getItem('phonoweave.language')" in HTML
    assert 'split_recommended' in HTML
    assert '建议区分录制' in HTML
