import pytest
from gpt.utils import check_tokenizer, file_sha256


def test_hash_tracks_content(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("tokenizer one"); b.write_text("tokenizer two")
    assert file_sha256(a) == file_sha256(a) and file_sha256(a) != file_sha256(b)


def test_check_tokenizer(tmp_path, capsys):
    a = tmp_path / "a.json"; a.write_text("x")
    check_tokenizer(a, file_sha256(a))                           # matches: silent
    with pytest.raises(ValueError):
        check_tokenizer(a, "0" * 64)                             # mismatch: error
    check_tokenizer(a, None)                                     # unknown: warning only
    assert "warning" in capsys.readouterr().out