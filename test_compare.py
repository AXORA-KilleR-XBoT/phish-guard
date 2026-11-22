import math

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import compare_models
from compare_models import candidates, check_folds, compare, main
from phishguard import SEED, PhishGuardError
from testdata import rows, write_tsv


def test_every_candidate_can_fit_and_predict():
    texts = ["free prize click now", "see you at lunch"] * 3
    labels = ["spam", "ham"] * 3
    for name, model in candidates().items():
        assert list(model.fit(texts, labels).predict(["free prize"])) == ["spam"], name


def test_scores_are_real_numbers_between_0_and_1_not_nan(tmp_path):
    # Regression test: with string labels and the default scorers every score used to be NaN.
    data = write_tsv(tmp_path / "d.tsv", rows(30, 30))
    results = compare(data, folds=3)
    assert set(results) == {"naive_bayes", "logistic_regression", "linear_svm"}
    for name, metrics in results.items():
        assert set(metrics) == {"precision", "recall", "f1"}
        for mean, std in metrics.values():
            assert not math.isnan(mean) and not math.isnan(std), name
            assert 0.0 <= mean <= 1.0 and std >= 0.0, name


def test_metrics_are_computed_for_the_spam_class(tmp_path, monkeypatch):
    """A model that always says 'ham' must score 0 for spam. (With 'ham' as the positive class it would score 1.)"""
    monkeypatch.setattr(compare_models, "candidates", lambda: {
        "always_ham": make_pipeline(DummyClassifier(strategy="constant", constant="ham")),
        "always_spam": make_pipeline(DummyClassifier(strategy="constant", constant="spam")),
    })
    results = compare(write_tsv(tmp_path / "d.tsv", rows(30, 30)), folds=3)
    assert results["always_ham"]["recall"][0] == 0.0 and results["always_ham"]["precision"][0] == 0.0
    assert results["always_ham"]["f1"][0] == 0.0
    assert results["always_spam"]["recall"][0] == 1.0
    assert results["always_spam"]["precision"][0] == pytest.approx(0.5)      # half of all messages are spam
    assert results["always_spam"]["f1"][0] == pytest.approx(2 / 3)


def test_a_failing_model_stops_the_run_instead_of_giving_nan(tmp_path, monkeypatch):
    def explode(texts):
        raise ZeroDivisionError("broken model")

    monkeypatch.setattr(compare_models, "candidates", lambda: {
        "broken": make_pipeline(FunctionTransformer(explode), DummyClassifier()),
    })
    with pytest.raises(ZeroDivisionError):
        compare(write_tsv(tmp_path / "d.tsv", rows(30, 30)), folds=3)


# ---------- validating the number of folds ----------

@pytest.mark.parametrize("folds", [0, 1, -3])
def test_fewer_than_two_folds_is_rejected(folds):
    with pytest.raises(PhishGuardError, match="at least 2"):
        check_folds(["spam"] * 10 + ["ham"] * 10, folds)


def test_more_folds_than_the_smallest_class_is_rejected():
    labels = ["spam"] * 3 + ["ham"] * 50
    with pytest.raises(PhishGuardError, match="smallest class has only 3 messages"):
        check_folds(labels, 5)
    assert check_folds(labels, 3) == 3                                       # exactly as many folds as messages: allowed


def test_compare_validates_folds_before_doing_any_work(tmp_path):
    data = write_tsv(tmp_path / "d.tsv", rows(4, 40))
    with pytest.raises(PhishGuardError, match="smallest class has only 4 messages"):
        compare(data, folds=5)


def test_every_fold_contains_both_classes_whenever_the_folds_are_accepted():
    from sklearn.model_selection import StratifiedKFold
    for n_spam in range(2, 12):
        for n_ham in range(2, 12):
            labels = ["spam"] * n_spam + ["ham"] * n_ham
            for folds in range(2, 12):
                try:
                    check_folds(labels, folds)
                except PhishGuardError:
                    continue
                for train, test in StratifiedKFold(folds, shuffle=True, random_state=42).split(labels, labels):
                    assert {labels[i] for i in test} == {labels[i] for i in train} == {"spam", "ham"}


def test_repeated_messages_are_removed_before_cross_validation(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(20, 20) * 4)                  # 160 lines, 40 different messages
    compare(data, folds=4)
    assert "40 messages (20 spam, 20 ham) after removing 120 repeated" in capsys.readouterr().out


# ---------- command line ----------

def test_cli_prints_one_line_per_model(tmp_path, capsys):
    data = write_tsv(tmp_path / "d.tsv", rows(30, 30))
    assert main(["--data", str(data), "--folds", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[2:]] == ["naive_bayes", "logistic_regression", "linear_svm"]
    assert all("nan" not in line.lower() for line in lines)


def test_cli_warns_about_small_datasets(tmp_path, capsys):
    assert main(["--data", str(write_tsv(tmp_path / "d.tsv", rows(10, 10))), "--folds", "5"]) == 0
    assert "rough hint" in capsys.readouterr().err


def test_cli_reports_problems_as_errors(tmp_path, capsys):
    good = write_tsv(tmp_path / "good.tsv", rows(30, 30))
    assert main(["--data", str(good), "--folds", "1"]) == 2
    assert main(["--data", str(good), "--folds", "31"]) == 2
    assert main(["--data", str(tmp_path / "missing.tsv")]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 3 and "Traceback" not in err


def test_cli_rejects_a_folds_value_that_is_not_a_number():
    with pytest.raises(SystemExit) as exc:
        main(["--folds", "many"])
    assert exc.value.code == 2


def test_results_are_repeatable(tmp_path):
    data = write_tsv(tmp_path / "d.tsv", rows(30, 30))
    assert compare(data, folds=3) == compare(data, folds=3)


def test_folds_are_shuffled_with_a_fixed_seed(tmp_path, monkeypatch):
    """Two runs on the same data must use the same folds, so the printed numbers can be reproduced."""
    seen = {}
    real = compare_models.StratifiedKFold

    def spy(**options):
        seen.update(options)
        return real(**options)

    monkeypatch.setattr(compare_models, "StratifiedKFold", spy)
    compare(write_tsv(tmp_path / "d.tsv", rows(30, 30)), folds=3)
    assert seen == {"n_splits": 3, "shuffle": True, "random_state": SEED}
