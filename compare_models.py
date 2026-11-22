"""Compare candidate models with stratified cross-validation (spam is the positive class).

Cross-validation splits the data into folds; each fold is used once as unseen test data while the
model is trained on the other folds. The scores are averaged over the folds. This script only
compares models: it does not save anything, and the numbers are not a promise about real-world use.
"""
import argparse
import sys
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from phishguard import LABELS, MIN_RELIABLE, POSITIVE, SEED, PhishGuardError, build_model, prepare

# The labels are the strings "spam" and "ham", so every metric must be told which one is the
# positive class. The default (pos_label=1) does not exist here and would give NaN scores.
SCORING = {
    "precision": make_scorer(precision_score, pos_label=POSITIVE, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=POSITIVE, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=POSITIVE, zero_division=0),
}


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def candidates():
    return {
        "naive_bayes": make_pipeline(tfidf(), MultinomialNB()),
        "logistic_regression": build_model(),
        "linear_svm": make_pipeline(tfidf(), LinearSVC(class_weight="balanced")),
    }


def check_folds(labels, folds):
    """Every fold must contain both classes, otherwise precision and recall are not defined."""
    counts = Counter(labels)
    smallest = min(counts[name] for name in LABELS)
    if folds < 2:
        raise PhishGuardError("--folds must be at least 2")
    if folds > smallest:
        raise PhishGuardError(
            f"--folds is {folds}, but the smallest class has only {smallest} messages. "
            f"Use at most {smallest} folds, or add more data"
        )
    return smallest


def compare(data, folds=5):
    """Print and return {model name: {metric: (mean, standard deviation over folds)}}."""
    texts, labels, removed = prepare(data)  # repeated messages are removed first, so folds cannot share a message
    smallest = check_folds(labels, folds)
    counts = Counter(labels)
    print(f"{len(texts)} messages ({counts['spam']} spam, {counts['ham']} ham) after removing "
          f"{removed['duplicates']} repeated and {removed['conflicting']} conflicting messages.")
    print(f"{folds}-fold stratified cross-validation; scores are for the '{POSITIVE}' class "
          "(mean +/- standard deviation over the folds).")
    if smallest // folds < MIN_RELIABLE:
        print(f"warning: fewer than {MIN_RELIABLE} messages of the smaller class per fold, so these scores "
              "are only a rough hint.", file=sys.stderr)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    results = {}
    for name, model in candidates().items():
        # The vectorizer is part of the pipeline, so it is fitted again inside every fold (no leakage).
        # error_score="raise": a failing fold must stop the run, not turn into a silent NaN.
        scores = cross_validate(model, texts, labels, cv=cv, scoring=SCORING, error_score="raise")
        results[name] = {metric: (float(scores[f"test_{metric}"].mean()), float(scores[f"test_{metric}"].std()))
                         for metric in SCORING}
        print(f"{name:<20} " + "  ".join(f"{metric}={mean:.3f}+/-{std:.3f}"
                                         for metric, (mean, std) in results[name].items()))
    return results


def main(argv=None):
    """Exit codes: 0 = done, 2 = error (bad dataset, impossible number of folds)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/sample_sms.tsv")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)
    try:
        compare(args.data, args.folds)
    except (PhishGuardError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
