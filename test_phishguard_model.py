import hashlib
import os
import pickle
import stat

import pytest

import phishguard
from phishguard import PhishGuardError, atomic_write, checksum_path, load_model, predict, save_model


def write_with_checksum(path, data):
    """Put raw bytes in `path` and a matching checksum file next to it."""
    path.write_bytes(data)
    checksum_path(path).write_text(f"{hashlib.sha256(data).hexdigest()}  {path.name}\n")


# ---------- saving and loading ----------

def test_round_trip_gives_the_same_predictions(model_file, sample_model):
    loaded = load_model(model_file)
    for text in ("free prize now", "see you tomorrow at lunch"):
        assert predict(loaded, text) == predict(sample_model, text)


def test_load_accepts_strings_and_paths(model_file):
    assert load_model(str(model_file)).classes_.tolist() == ["ham", "spam"]


def test_checksum_file_matches_the_sha256sum_format(model_file):
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert checksum_path(model_file).read_text() == f"{digest}  model.joblib\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_saved_files_are_private_to_their_owner(model_file):
    assert stat.S_IMODE(model_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(checksum_path(model_file).stat().st_mode) == 0o600


def test_save_creates_missing_folders(tmp_path, sample_model):
    path = tmp_path / "a" / "b" / "model.joblib"
    save_model(sample_model, path)
    assert load_model(path)


# ---------- refusing files that cannot be trusted ----------

def test_missing_model_file(tmp_path):
    with pytest.raises(PhishGuardError, match="Cannot read model file"):
        load_model(tmp_path / "missing.joblib")


def test_missing_checksum_file_means_refusal(model_file):
    checksum_path(model_file).unlink()
    with pytest.raises(PhishGuardError, match="checksum file"):
        load_model(model_file)


@pytest.mark.parametrize("content", ["", "\n", "not-a-hash  model.joblib\n", "AB" * 32 + "\n", "ab" * 31 + "\n"])
def test_malformed_checksum_file_means_refusal(model_file, content):
    checksum_path(model_file).write_text(content)
    with pytest.raises(PhishGuardError, match="checksum"):
        load_model(model_file)


def test_a_changed_model_file_is_refused(model_file):
    data = bytearray(model_file.read_bytes())
    data[len(data) // 2] ^= 0xFF                                   # flip one byte in the middle
    model_file.write_bytes(bytes(data))
    with pytest.raises(PhishGuardError, match="does not match its checksum"):
        load_model(model_file)


def test_a_truncated_model_file_is_refused(model_file):
    model_file.write_bytes(model_file.read_bytes()[:-10])
    with pytest.raises(PhishGuardError, match="does not match its checksum"):
        load_model(model_file)


def test_unpickling_never_happens_for_a_file_without_a_matching_checksum(tmp_path):
    """The reason for the checksum: a hostile pickle must not get to run any of its code."""
    marker = tmp_path / "pwned.txt"

    class Hostile:
        def __reduce__(self):
            return (open, (str(marker), "w"))                      # harmless stand-in for "run my code"

    path = tmp_path / "model.joblib"
    path.write_bytes(pickle.dumps(Hostile()))
    with pytest.raises(PhishGuardError, match="checksum"):         # no checksum file at all
        load_model(path)
    checksum_path(path).write_text("0" * 64 + "\n")                # a checksum that does not match
    with pytest.raises(PhishGuardError, match="does not match"):
        load_model(path)
    assert not marker.exists()


def test_the_attack_the_checksum_is_for_really_does_run_code_without_it(tmp_path):
    """Control for the test above: plain unpickling of the same bytes DOES create the marker."""
    marker = tmp_path / "pwned.txt"

    class Hostile:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    pickle.loads(pickle.dumps(Hostile()))
    assert marker.exists()


# ---------- files that match their checksum but are not usable models ----------

def test_damaged_pickle_data_is_a_clean_error(tmp_path):
    path = tmp_path / "model.joblib"
    write_with_checksum(path, b"this is not a pickle at all")
    with pytest.raises(PhishGuardError, match="damaged or was made with other library versions"):
        load_model(path)


def test_a_pickle_that_needs_a_missing_library_is_a_clean_error(tmp_path):
    path = tmp_path / "model.joblib"
    write_with_checksum(path, b"cno_such_module_xyz\nThing\n.")     # a pickle that imports a module that does not exist
    with pytest.raises(PhishGuardError, match="damaged or was made with other library versions"):
        load_model(path)


@pytest.mark.parametrize("wrong", [{"a": 1}, [1, 2, 3], "text", None, 42])
def test_valid_pickles_of_the_wrong_type_are_refused(tmp_path, wrong):
    path = tmp_path / "model.joblib"
    write_with_checksum(path, pickle.dumps(wrong))
    with pytest.raises(PhishGuardError, match="not a phish-guard model"):
        load_model(path)


def test_a_model_for_other_labels_is_refused(tmp_path):
    other = phishguard.build_model().fit(["meow purr", "woof bark", "meow purr nap", "bark woof fetch"], ["cat", "dog", "cat", "dog"])
    path = tmp_path / "model.joblib"
    save_model(other, path)
    with pytest.raises(PhishGuardError, match="not a phish-guard model"):
        load_model(path)


def test_an_unfitted_model_is_refused(tmp_path):
    path = tmp_path / "model.joblib"
    save_model(phishguard.build_model(), path)
    with pytest.raises(PhishGuardError, match="not a phish-guard model"):
        load_model(path)


def test_the_check_command_refuses_untrusted_files_with_exit_code_2(model_file, capsys):
    model_file.write_bytes(model_file.read_bytes() + b"x")
    assert phishguard.main(["check", "hello", "--model", str(model_file)]) == 2
    err = capsys.readouterr().err
    assert "Refusing to load" in err and "Traceback" not in err


# ---------- atomic writing ----------

def test_failed_replace_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "file.bin"
    atomic_write(path, b"old")

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(path, b"new")
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


def test_an_interrupted_save_fails_closed(tmp_path, sample_model, monkeypatch):
    """If the model was replaced but its checksum could not be, the pair no longer matches and loading is refused."""
    path = tmp_path / "model.joblib"
    save_model(sample_model, path)
    other = phishguard.build_model().fit(["free prize", "see you", "win cash", "lunch time"], ["spam", "ham", "spam", "ham"])
    real = phishguard.atomic_write
    calls = []

    def fail_on_the_checksum(target, data):
        calls.append(target)
        if len(calls) == 2:
            raise OSError(5, "Input/output error")
        real(target, data)

    monkeypatch.setattr(phishguard, "atomic_write", fail_on_the_checksum)
    with pytest.raises(OSError):
        save_model(other, path)
    monkeypatch.undo()
    with pytest.raises(PhishGuardError, match="does not match its checksum"):
        load_model(path)


def test_failure_while_writing_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "file.bin"
    atomic_write(path, b"old")

    def boom(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", boom)                         # the data must be forced to disk before the rename
    with pytest.raises(OSError):
        atomic_write(path, b"new")
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]
