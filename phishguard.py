"""phish-guard: an educational SMS spam classifier (TF-IDF + logistic regression).

It labels one short text message as "spam" or "ham" (ham = not spam). It is NOT a complete
phishing detector: it learns from SMS spam examples only, it never opens or checks URLs, and a
low spam score is not proof that a message is safe.
"""
import argparse
import hashlib
import io
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

LABELS = ("ham", "spam")
POSITIVE = "spam"      # the class that precision / recall / F1 are reported for, in every script
TEST_SIZE = 0.25       # share of the data kept aside to grade the model
SEED = 42              # fixed so that two runs on the same data give the same split
MAX_TEXT_CHARS = 10_000
MIN_RELIABLE = 20      # fewer test messages than this for a class: the scores are only a rough hint


class PhishGuardError(ValueError):
    """A problem with the dataset, the model file or the input, shown to the user as a short message."""


# ---------- dataset ----------

def load(path):
    """Read 'label<TAB>message' lines (the UCI SMS Spam Collection layout, no header row).

    Returns (texts, labels). Blank lines are skipped; anything else that does not fit raises
    PhishGuardError, because silently skipping bad rows would hide a broken dataset.
    """
    texts, labels = [], []
    try:
        with open(path, encoding="utf-8-sig") as f:  # utf-8-sig also accepts a leading byte-order mark
            for line_number, line in enumerate(f, start=1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                label, _, message = line.partition("\t")  # a line without a tab gives an empty message
                if label not in LABELS or not message.strip():
                    raise PhishGuardError(
                        f"Invalid dataset row {line_number}: expected 'spam<TAB>message' "
                        "or 'ham<TAB>message' (and no header row)"
                    )
                labels.append(label)
                texts.append(message)
    except UnicodeDecodeError as exc:
        raise PhishGuardError("The dataset is not valid UTF-8 text (is it saved in another encoding?)") from exc
    if not texts:
        raise PhishGuardError("Dataset is empty")
    if set(labels) != set(LABELS):
        raise PhishGuardError("Dataset must contain both 'spam' and 'ham' labels")
    return texts, labels


def normalize(text):
    """The form of a message used to decide whether two messages are the same."""
    return " ".join(text.split()).casefold()


def deduplicate(texts, labels):
    """Drop repeated messages. Returns (texts, labels, removed).

    The SMS data contains many exact repeats. If one copy lands in the training set and another in
    the test set, the model is graded on a message it has already seen and every score looks better
    than it is. Messages are compared after lower-casing and collapsing white space. A message that
    appears with BOTH labels is ambiguous, so all of its copies are dropped.
    """
    labels_of = {}
    for text, label in zip(texts, labels):
        labels_of.setdefault(normalize(text), set()).add(label)
    kept_texts, kept_labels, seen = [], [], set()
    removed = {"duplicates": 0, "conflicting": 0}
    for text, label in zip(texts, labels):
        key = normalize(text)
        if len(labels_of[key]) > 1:
            removed["conflicting"] += 1
        elif key in seen:
            removed["duplicates"] += 1
        else:
            seen.add(key)
            kept_texts.append(text)
            kept_labels.append(label)
    return kept_texts, kept_labels, removed


def prepare(path):
    """Load a dataset and remove repeated messages. Returns (texts, labels, removed)."""
    texts, labels, removed = deduplicate(*load(path))
    counts = Counter(labels)
    for name in LABELS:
        if counts[name] == 0:
            raise PhishGuardError(f"No '{name}' messages are left after removing duplicates and conflicting labels")
    return texts, labels, removed


def split(texts, labels):
    """Stratified train/test split (both sets keep the spam/ham ratio). Returns x_tr, x_te, y_tr, y_te.

    Raises PhishGuardError, with a plain explanation, when the data is too small for a fair split.
    """
    counts = Counter(labels)
    n_test = math.ceil(TEST_SIZE * len(labels))
    for name in LABELS:
        if counts[name] < 2:
            raise PhishGuardError(
                f"Not enough data: only {counts[name]} '{name}' message(s). Each class needs at least 2, "
                "so that both the training set and the test set can contain it"
            )
    if n_test < len(LABELS) or len(labels) - n_test < len(LABELS):
        raise PhishGuardError(
            f"Not enough data: {len(labels)} messages are too few for a {TEST_SIZE:.0%} test set "
            f"that holds at least {len(LABELS)} messages"
        )
    x_tr, x_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )
    # With very small classes the random split can still leave one class out of a set.
    for set_name, part in (("training", y_tr), ("test", y_te)):
        for name in LABELS:
            if name not in part:
                raise PhishGuardError(
                    f"Not enough data: the {set_name} set ended up with no '{name}' messages. "
                    f"Add more '{name}' messages"
                )
    return x_tr, x_te, y_tr, y_te


# ---------- model ----------

def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def check_model(model):
    """Raise PhishGuardError unless `model` is a fitted classifier for the 'ham' and 'spam' classes."""
    try:
        usable = callable(model.predict_proba) and set(model.classes_) == set(LABELS)
    except Exception:  # not a fitted classifier at all
        usable = False
    if not usable:
        raise PhishGuardError("This is not a phish-guard model (a fitted classifier with 'ham' and 'spam' classes)")


def checksum_path(model_path):
    return Path(str(model_path) + ".sha256")


def atomic_write(path, data):
    """Write bytes in one step: a temporary file next to `path` is renamed over it,
    so an interrupted write never leaves a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # private (0600) by default
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_model(model, path):
    """Save the model and, next to it, a '<path>.sha256' file in the format `sha256sum -c` understands."""
    path = Path(path)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    data = buffer.getvalue()
    atomic_write(path, data)
    atomic_write(checksum_path(path), f"{hashlib.sha256(data).hexdigest()}  {path.name}\n".encode("ascii"))


def load_model(path):
    """Load a model written by save_model(), after checking it against its checksum file.

    A joblib file is a Python pickle, and unpickling a file can run arbitrary code. So the file is
    read once, hashed, and only unpickled when the hash equals the recorded one: a file that was
    swapped, edited or cut short is refused BEFORE any of its code runs. This is an integrity check,
    not authentication: someone who can replace both files can still fool it. Only load models
    that you created yourself, and keep the folder with the model writable only by you.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PhishGuardError(f"Cannot read model file {path}: {exc.strerror or exc}") from exc
    try:
        recorded = checksum_path(path).read_text(encoding="ascii").split()[0]
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        raise PhishGuardError(
            f"Refusing to load {path}: its checksum file {checksum_path(path).name} is missing or unreadable. "
            "Create the model with 'phishguard.py train' so that the checksum is written"
        ) from exc
    if recorded != hashlib.sha256(data).hexdigest():  # also rejects anything that is not a lower-case SHA-256
        raise PhishGuardError(f"Refusing to load {path}: it does not match its checksum file (changed or damaged?)")
    try:
        model = joblib.load(io.BytesIO(data))
    except Exception as exc:  # damaged pickle, or a class from a library version that is not installed
        raise PhishGuardError(
            f"Cannot load {path}: the file is damaged or was made with other library versions"
        ) from exc
    check_model(model)
    return model


def predict(model, text):
    """Return (label, spam_probability) for one message.

    The probability is a model score, not a guarantee: a low score does not prove a message is safe.
    """
    if not isinstance(text, str) or not text.strip():
        raise PhishGuardError("text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise PhishGuardError(f"text must not be longer than {MAX_TEXT_CHARS} characters")
    check_model(model)
    try:
        spam_prob = float(model.predict_proba([text])[0][list(model.classes_).index(POSITIVE)])
    except Exception as exc:  # for example a model made with an incompatible scikit-learn version
        raise PhishGuardError(
            "The model could not score this message (was it made with other library versions? "
            "Train it again with 'phishguard.py train')"
        ) from exc
    if not 0.0 <= spam_prob <= 1.0:  # also catches NaN
        raise PhishGuardError("The model returned an invalid probability")
    return (POSITIVE if spam_prob >= 0.5 else "ham"), spam_prob


# ---------- command line ----------

def train(data, model_path):
    """Evaluate on held-out data, then save a model trained on all messages."""
    texts, labels, removed = prepare(data)
    counts = Counter(labels)
    print(f"Dataset: {len(texts)} messages ({counts['spam']} spam, {counts['ham']} ham). Removed "
          f"{removed['duplicates']} repeated and {removed['conflicting']} conflicting messages first, "
          "so that no message can be in both the training and the test set.")
    x_tr, x_te, y_tr, y_te = split(texts, labels)
    held_out = build_model().fit(x_tr, y_tr)
    print(f"\nHeld-out evaluation: trained on {len(x_tr)} messages, tested on {len(x_te)} messages it never saw")
    print(classification_report(y_te, held_out.predict(x_te), labels=list(LABELS), zero_division=0))
    thin = [name for name in LABELS if y_te.count(name) < MIN_RELIABLE]
    if thin:
        print(f"warning: the test set has fewer than {MIN_RELIABLE} {' and '.join(thin)} messages, so "
              "these scores are only a rough hint. Use a bigger dataset for real results.", file=sys.stderr)
    final = build_model().fit(texts, labels)
    save_model(final, model_path)
    print(f"Saved model to {model_path} (checksum: {checksum_path(model_path)}).")
    print("The saved model is retrained on all messages, so it has no held-out score of its own; "
          "the report above belongs to the model trained on the training split only.")


def main(argv=None):
    """Exit codes: 0 = done, 2 = error (bad dataset, unusable model file, bad input)."""
    p = argparse.ArgumentParser(description="Educational SMS spam classifier (not a complete phishing detector)")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="evaluate on held-out data, then save a model")
    t.add_argument("--data", default="data/sample_sms.tsv")
    t.add_argument("--model", default="model.joblib")
    c = sub.add_parser("check", help="classify one message")
    c.add_argument("text")
    c.add_argument("--model", default="model.joblib")
    args = p.parse_args(argv)
    try:
        if args.cmd == "train":
            train(args.data, args.model)
        else:
            label, prob = predict(load_model(args.model), args.text)
            print(f"{label} (spam probability: {prob:.0%})")
            if label == "ham":
                print("A low spam score is not proof that a message is safe.")
    except (PhishGuardError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
