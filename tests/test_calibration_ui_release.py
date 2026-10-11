"""The standard calibration labeler does not expose legacy-label migration."""
from phonoweave.calibration_manual_labeler import HTML, _Handler


def test_standard_labeler_has_no_legacy_copy_ui():
    assert 'id="copyLegacy"' not in HTML
    assert '/copy-legacy' not in HTML
    assert '复制旧标记到当前类型' not in HTML


def test_standard_labeler_does_not_expose_legacy_import_action():
    import inspect
    source = inspect.getsource(_Handler.do_POST)
    assert '/copy-legacy' not in source
    assert 'copy_legacy_label' not in source
