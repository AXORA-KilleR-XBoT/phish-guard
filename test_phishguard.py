from pathlib import Path

import pytest

import phishguard
from phishguard import MAX_TEXT_CHARS, PhishGuardError, deduplicate, load, normalize, predict, prepare, split
from testdata import rows, write_tsv


# ---------- loading a dataset ----------

def test_load_reads_labels_and_messages(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_bytes(b"ham\thello\r\n\n   \nspam\tfree\tprize\n")        # CRLF, blank lines, a tab inside a message
    assert load(f) == (["hello", "free\tprize"], ["ham", "spam"])


def test_load_accepts_a_byte_order_mark(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_bytes(b"\xef\xbb\xbfham\thello\nspam\tfree prize\n")
    assert load(f) == (["hello", "free prize"], ["ham", "spam"])


@pytest.mark.parametrize("content, bad_row", [
    ("ham\thello\nspam free prize\n", 2),               # a space instead of a tab
    ("ham\thello\nunknown\tbad label\nspam\tfree\n", 2),
    ("ham\thello\nSpam\tfree\n", 2),                    # labels are case-sensitive
    ("label\tmessage\nham\thello\nspam\tfree\n", 1),    # header row
    ("ham\thello\nspam\t\n", 2),                        # empty message
    ("ham\thello\nspam\t   \n", 2),                     # blank message
    ("ham\thello\n\tno label\n", 2),
    ("ham\n", 1),                                       # no tab at all
])
def test_load_rejects_malformed_rows_and_names_the_row(tmp_path, content, bad_row):
    f = tmp_path / "bad.tsv"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(PhishGuardError, match=f"Invalid dataset row {bad_row}:"):
        load(f)


@pytest.mark.parametrize("content, message", [
    ("", "empty"),
    ("\n\n  \n", "empty"),
    ("ham\ta\nham\tb\n", "both"),
    ("spam\ta\nspam\tb\n", "both"),
])
def test_load_rejects_empty_and_one_class_datasets(tmp_path, content, message):
    f = tmp_path / "d.tsv"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(PhishGuardError, match=message):
        load(f)


def test_load_rejects_text_that_is_not_utf8(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_bytes(b"ham\tcaf\xe9\nspam\tfree\n")                 # Latin-1 bytes
    with pytest.raises(PhishGuardError, match="UTF-8"):
        load(f)


def test_load_reports_a_missing_file_as_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.tsv")
    with pytest.raises(OSError):
        load(tmp_path)                                           # a folder


def test_the_sample_dataset_in_the_repository_is_valid():
    texts, labels = load(Path(__file__).parent / "data" / "sample_sms.tsv")
    assert len(texts) == 20 and sorted(set(labels)) == ["ham", "spam"]


# ---------- repeated messages (leakage) ----------

def test_deduplicate_keeps_the_first_copy_and_the_order():
    texts = ["a", "b", "a", "c", "b", "a"]
    labels = ["ham", "spam", "ham", "ham", "spam", "ham"]
    assert deduplicate(texts, labels) == (["a", "b", "c"], ["ham", "spam", "ham"],
                                          {"duplicates": 3, "conflicting": 0})


def test_deduplicate_ignores_case_and_white_space():
    texts = ["Free  PRIZE now", "free prize now", " free\tprize now\n", "something else"]
    kept, _, removed = deduplicate(texts, ["spam"] * 4)
    assert kept == ["Free  PRIZE now", "something else"] and removed["duplicates"] == 2


def test_deduplicate_drops_every_copy_of_a_message_with_conflicting_labels():
    texts = ["same", "other", "SAME", "same"]
    labels = ["spam", "ham", "ham", "spam"]
    assert deduplicate(texts, labels) == (["other"], ["ham"], {"duplicates": 0, "conflicting": 3})


def test_normalize():
    assert normalize("  Hello\t WORLD \n") == "hello world"


def test_prepare_refuses_data_that_loses_a_whole_class(tmp_path):
    f = write_tsv(tmp_path / "d.tsv", [("spam", "x"), ("ham", "x"), ("ham", "y"), ("ham", "z")])
    with pytest.raises(PhishGuardError, match="No 'spam' messages are left"):
        prepare(f)                                               # the only spam message also appears as ham


def test_no_message_is_in_both_the_training_and_the_test_set(tmp_path):
    pairs = rows(30, 30) * 3                                     # every message three times
    f = write_tsv(tmp_path / "d.tsv", pairs)
    texts, labels, removed = prepare(f)
    assert removed["duplicates"] == 120 and len(texts) == 60
    x_tr, x_te, _, _ = split(texts, labels)
    assert {normalize(t) for t in x_tr}.isdisjoint({normalize(t) for t in x_te})


# ---------- splitting: small and unbalanced data ----------

def test_four_messages_are_not_enough_even_though_the_old_check_allowed_them():
    # The original code only required 4 messages and then crashed inside scikit-learn.
    labels = ["spam", "spam", "ham", "ham"]
    with pytest.raises(PhishGuardError, match="Not enough data"):
        split(["a", "b", "c", "d"], labels)


@pytest.mark.parametrize("n_spam, n_ham, name", [(1, 9, "spam"), (9, 1, "ham"), (0, 9, "spam"), (9, 0, "ham")])
def test_a_class_with_fewer_than_two_messages_is_rejected(n_spam, n_ham, name):
    pairs = rows(n_spam, n_ham)
    with pytest.raises(PhishGuardError, match=f"'{name}'"):
        split([t for _, t in pairs], [label for label, _ in pairs])


def test_every_small_class_size_gives_a_clear_error_or_a_valid_split():
    """No combination may crash with a raw scikit-learn error, or return a split that lacks a class."""
    for n_spam in range(0, 13):
        for n_ham in range(0, 13):
            pairs = rows(n_spam, n_ham)
            texts, labels = [t for _, t in pairs], [label for label, _ in pairs]
            try:
                x_tr, x_te, y_tr, y_te = split(texts, labels)
            except PhishGuardError:
                continue
            assert set(y_tr) == set(y_te) == {"spam", "ham"}, (n_spam, n_ham)
            assert len(x_tr) + len(x_te) == len(texts) and not set(x_tr) & set(x_te)


def test_a_split_that_leaves_a_class_out_of_the_test_set_is_refused(monkeypatch):
    def lopsided(texts, labels, **options):                       # what a bad random draw can look like
        return texts[:6], texts[6:], labels[:6], ["ham"] * (len(labels) - 6)

    monkeypatch.setattr(phishguard, "train_test_split", lopsided)
    pairs = rows(4, 4)
    with pytest.raises(PhishGuardError, match="test set ended up with no 'spam'"):
        split([t for _, t in pairs], [label for label, _ in pairs])


def test_split_is_stratified_and_repeatable():
    pairs = rows(20, 60)
    texts, labels = [t for _, t in pairs], [label for label, _ in pairs]
    first = split(texts, labels)
    assert first == split(texts, labels)
    assert first[3].count("spam") == 5 and first[3].count("ham") == 15     # the 1:3 ratio is kept in the test set


# ---------- training from the command line ----------

def train_args(tmp_path, data):
    return ["train", "--data", str(data), "--model", str(tmp_path / "out" / "model.joblib")]


def test_train_evaluates_on_held_out_data_and_saves_a_checked_model(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(100, 100))
    assert phishguard.main(train_args(tmp_path, data)) == 0
    out = capsys.readouterr()
    assert "Dataset: 200 messages (100 spam, 100 ham)" in out.out
    assert "trained on 150 messages, tested on 50 messages it never saw" in out.out
    assert "precision" in out.out and "recall" in out.out
    assert "retrained on all messages" in out.out and "no held-out score of its own" in out.out
    assert out.err == ""                                          # 25 test messages per class: nothing to warn about
    model = phishguard.load_model(tmp_path / "out" / "model.joblib")
    assert predict(model, "WINNER! Claim your free prize number 7 now, click the link")[0] == "spam"


def test_train_reports_how_many_repeated_messages_it_removed(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(60, 60) + rows(10, 0))
    assert phishguard.main(train_args(tmp_path, data)) == 0
    assert "Dataset: 120 messages (60 spam, 60 ham). Removed 10 repeated and 0 conflicting" in capsys.readouterr().out


def test_train_warns_when_the_test_set_is_too_small_to_trust(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(10, 10))
    assert phishguard.main(train_args(tmp_path, data)) == 0
    assert "only a rough hint" in capsys.readouterr().err


@pytest.mark.parametrize("pairs", [rows(2, 2), rows(1, 9), rows(3, 1), rows(0, 5)])
def test_train_refuses_tiny_datasets_with_a_clear_message_and_writes_nothing(tmp_path, capsys, pairs):
    data = write_tsv(tmp_path / "d.tsv", pairs)
    assert phishguard.main(train_args(tmp_path, data)) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err
    assert not (tmp_path / "out").exists()                        # no model, no checksum, nothing


def test_train_reports_bad_input_files_as_errors(tmp_path, capsys):
    assert phishguard.main(train_args(tmp_path, tmp_path / "missing.tsv")) == 2
    bad = tmp_path / "bad.tsv"
    bad.write_text("ham\thello\nbroken line\n")
    assert phishguard.main(train_args(tmp_path, bad)) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 2 and "Invalid dataset row 2" in err and "Traceback" not in err


# ---------- predicting ----------

def test_flags_obvious_spam(sample_model):
    label, prob = predict(sample_model, "URGENT! You won a FREE prize, click http://bit.ly/x now")
    assert label == "spam" and prob > 0.5


def test_keeps_a_normal_message(sample_model):
    label, prob = predict(sample_model, "Can we move lunch to tomorrow, after the meeting?")
    assert label == "ham" and prob < 0.5


def test_the_label_follows_the_probability_threshold(sample_model):
    for text in ("free prize now", "see you tomorrow", "click the link to claim", "thanks for yesterday"):
        label, prob = predict(sample_model, text)
        assert 0.0 <= prob <= 1.0 and label == ("spam" if prob >= 0.5 else "ham")


@pytest.mark.parametrize("bad", [None, 123, 1.5, ["free prize"], b"free prize", {"text": "x"}, "", "   ", "\n\t"])
def test_predict_rejects_input_that_is_not_a_non_empty_string(sample_model, bad):
    with pytest.raises(PhishGuardError, match="non-empty string"):
        predict(sample_model, bad)


def test_predict_limits_the_message_length(sample_model):
    assert predict(sample_model, "x" * MAX_TEXT_CHARS)                   # exactly at the limit is fine
    with pytest.raises(PhishGuardError, match="longer than"):
        predict(sample_model, "x" * (MAX_TEXT_CHARS + 1))


def test_predict_refuses_objects_that_are_not_phish_guard_models():
    class OnlyHam:
        classes_ = ["ham"]
        predict_proba = staticmethod(lambda texts: [[1.0]])

    class NoProbabilities:
        classes_ = ["ham", "spam"]

    for not_a_model in (None, object(), OnlyHam(), NoProbabilities(), phishguard.build_model()):   # the last: not fitted
        with pytest.raises(PhishGuardError, match="not a phish-guard model"):
            predict(not_a_model, "hello")


def test_predict_refuses_an_invalid_probability():
    class Broken:
        classes_ = ["ham", "spam"]
        predict_proba = staticmethod(lambda texts: [[float("nan"), float("nan")]])

    with pytest.raises(PhishGuardError, match="invalid probability"):
        predict(Broken(), "hello")


class ExplodingModel:
    """Looks like a model, but cannot score anything (for example, one made with another scikit-learn)."""
    classes_ = ["ham", "spam"]

    def predict_proba(self, texts):
        raise AttributeError("'LogisticRegression' object has no attribute 'multi_class'")


def test_predict_turns_a_failing_model_into_a_clear_error():
    with pytest.raises(PhishGuardError, match="could not score"):
        predict(ExplodingModel(), "hello")


# ---------- the check command ----------

def test_check_prints_label_and_probability(model_file, capsys):
    assert phishguard.main(["check", "URGENT! You won a FREE prize, click http://bit.ly/x now", "--model", str(model_file)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("spam (spam probability: ") and "not proof" not in out
    assert phishguard.main(["check", "Can we move lunch to tomorrow, after the meeting?", "--model", str(model_file)]) == 0
    assert "not proof that a message is safe" in capsys.readouterr().out      # a low score is never called "safe"


def test_check_reports_model_and_input_problems_as_errors(model_file, tmp_path, capsys):
    assert phishguard.main(["check", "hello", "--model", str(tmp_path / "missing.joblib")]) == 2
    assert phishguard.main(["check", "   ", "--model", str(model_file)]) == 2
    assert phishguard.main(["check", "x" * (MAX_TEXT_CHARS + 1), "--model", str(model_file)]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 3 and "Traceback" not in err


def test_check_reports_a_failing_model_as_an_error_not_a_traceback(model_file, monkeypatch, capsys):
    monkeypatch.setattr(phishguard, "load_model", lambda path: ExplodingModel())
    assert phishguard.main(["check", "hello", "--model", str(model_file)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err


def test_bad_usage_exits_with_2():
    for argv in ([], ["bogus"], ["check"], ["train", "--bogus"]):
        with pytest.raises(SystemExit) as exc:
            phishguard.main(argv)
        assert exc.value.code == 2


# ---------- the evaluation must be honest ----------

def test_the_held_out_model_never_sees_the_test_messages(tmp_path, monkeypatch, capsys):
    """Spy on every fit: the graded model is trained on the training split only, the saved model on everything."""
    from sklearn.pipeline import Pipeline
    fits = []
    original = Pipeline.fit

    def spy(self, X, y=None, **options):
        fits.append(list(X))
        return original(self, X, y, **options)

    monkeypatch.setattr(Pipeline, "fit", spy)
    data = write_tsv(tmp_path / "d.tsv", rows(100, 100))
    assert phishguard.main(train_args(tmp_path, data)) == 0
    texts, labels, _ = prepare(data)
    x_tr, x_te, _, _ = split(texts, labels)
    assert len(fits) == 2
    assert fits[0] == x_tr and set(fits[0]).isdisjoint(x_te)       # 1st fit: graded on messages it never saw
    assert fits[1] == texts                                         # 2nd fit: the saved model uses all the data


def test_no_warning_when_every_class_has_exactly_the_minimum_test_messages(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(80, 80))               # 25% test set = 20 spam + 20 ham
    assert phishguard.main(train_args(tmp_path, data)) == 0
    assert capsys.readouterr().err == ""


def test_a_probability_of_exactly_one_half_counts_as_spam():
    class Half:
        classes_ = ["ham", "spam"]
        predict_proba = staticmethod(lambda texts: [[0.5, 0.5]])

    assert predict(Half(), "hello") == ("spam", 0.5)
