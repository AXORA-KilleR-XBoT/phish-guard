# Phish Guard

An educational **SMS spam classifier**: a TF-IDF + logistic regression model trained on the public
SMS Spam Collection, a command line tool to train and use it, and a small FastAPI endpoint for local
experiments.

> **Please read this first.** Despite the name, this is **not** a comprehensive phishing detector. It was
> trained on SMS text messages labelled spam or not spam. It does not look at links, domains, web pages,
> sender reputation or attachments, and it knows nothing about languages or scams that are not in its
> training data. **A low score does not mean a message or a link is safe.** Do not use it to decide whether
> to open a link or share personal data.

<!-- Placeholders to replace before publishing (search for YOUR):
     - YOUR_GITHUB_USERNAME in the badge below (remove the comment markers after the repository exists)
     - YOUR NAME in LICENSE
[![CI](https://github.com/YOUR_GITHUB_USERNAME/phish-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/phish-guard/actions/workflows/tests.yml)
-->

## What problem it solves

It is a small, complete example of a machine learning workflow: get data, check it, evaluate a model
honestly, save it safely, and serve it. The checking and the honesty are the point. The accuracy of the
model on SMS spam is a side effect.

## Features

- Loads and validates a tab-separated `label<TAB>message` file, with clear errors for malformed files.
- Removes repeated messages **before** splitting, so the same text cannot be in both the training and the test set.
- Reports a held-out evaluation (precision, recall, F1 per class), then saves a model trained on all messages.
- `compare_models.py` compares three classifiers with stratified cross-validation, scored for the spam class.
- Saves the model with a SHA-256 checksum file and refuses to load a model that does not match it.
- `POST /check` API with input validation, errors that never echo the message or leak file paths, and a
  refusal to start when the model is missing or damaged.
- `download_data.py` fetches the data set from a fixed commit and verifies a pinned SHA-256 before saving it.
- Development Dockerfile (non-root user, model trained while the image is built).

## How it works

```text
sms.tsv --> load + validate --> remove repeated messages --> stratified split (75 / 25)
        --> TF-IDF (1-2 word n-grams) + logistic regression --> held-out report
        --> retrain on all messages --> model.joblib + model.joblib.sha256 --> CLI check / API /check
```

| File | Purpose |
| --- | --- |
| `phishguard.py` | Data loading, model, training, safe model loading, prediction and the `train` / `check` commands. |
| `compare_models.py` | Cross-validated comparison of Naive Bayes, logistic regression and linear SVM. |
| `api.py` | The FastAPI app (`create_app()` builds it; the model is loaded when the server starts). |
| `download_data.py` | Verified download of the data set. |
| `data/sample_sms.tsv` | A 20-message toy file (10 spam, 10 ham): the default for `--data`, used by one test and by the CI smoke run. Its scores mean nothing. |
| `Dockerfile` | Development container image for the API. |

## Requirements

- **Python 3.11 or newer** (scikit-learn 1.9 does not support Python 3.10). Tested locally on 3.11, 3.12 and 3.13 on Linux.
- Internet access once, to download the data set (about 480 KB).

## Quick start

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python phishguard.py train --data data/sms.tsv --model model.joblib
python phishguard.py check "URGENT! You have won a prize. Call now to claim it" --model model.joblib
```

Windows (PowerShell). These commands were not run while preparing this project:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python phishguard.py train --data data/sms.tsv --model model.joblib
python phishguard.py check "URGENT! You have won a prize. Call now to claim it" --model model.joblib
```

No internet? `python phishguard.py train --data data/sample_sms.tsv --model model.joblib` works with the
toy file, but its scores mean nothing (the program warns about this).

## Example output

Real output of the commands above on the full data set (Python 3.12, scikit-learn 1.9.1):

```text
$ python phishguard.py train --data data/sms.tsv --model model.joblib
Dataset: 5159 messages (642 spam, 4517 ham). Removed 415 repeated and 0 conflicting messages first, so that no message can be in both the training and the test set.

Held-out evaluation: trained on 3869 messages, tested on 1290 messages it never saw
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99      1129
        spam       0.92      0.94      0.93       161

    accuracy                           0.98      1290
   macro avg       0.96      0.96      0.96      1290
weighted avg       0.98      0.98      0.98      1290

Saved model to model.joblib (checksum: model.joblib.sha256).
The saved model is retrained on all messages, so it has no held-out score of its own; the report above belongs to the model trained on the training split only.

$ python phishguard.py check "Are we still meeting for lunch tomorrow?" --model model.joblib
ham (spam probability: 12%)
A low spam score is not proof that a message is safe.

$ python phishguard.py check "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry" --model model.joblib
spam (spam probability: 82%)

$ python phishguard.py check "Your bank account is locked. Verify your identity at http://secure-login.example/verify" --model model.joblib
spam (spam probability: 51%)
```

The last message is a typical phishing text, and the model is close to undecided about it. That is the
limitation described at the top, not a bug: the training data is mostly prize and promotion spam.

## Data set and credits

The data is the **SMS Spam Collection** by Tiago A. Almeida and José María Gómez Hidalgo (UCI Machine
Learning Repository), licensed **CC BY 4.0**, which requires credit:

> Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

`download_data.py` does not use the UCI site. It downloads a public copy kept by Kevin Markham
(`justmarkham/pycon-2016-tutorial`) from one fixed commit and saves the file only if its SHA-256 is
`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`. The copy has 5,574 messages, 4,827 ham
and 747 spam, the numbers published for the original. It was **not** compared with the UCI file message by
message. The data is not stored in this repository (`data/sms.tsv` is git-ignored).

## Model evaluation

- **Repeated messages are removed first.** The raw file contains 415 repeated messages. If they stay in,
  a copy of a test message can sit in the training set and the scores look better than they are. Measured
  here with 5-fold cross-validation (spam F1): naive Bayes 0.768 with repeats against 0.671 without,
  logistic regression 0.943 against 0.930, linear SVM 0.953 against 0.942. A message that appears with
  conflicting labels is dropped entirely (this data set has none).
- **Held-out vs. final model.** `train` evaluates on 25 % of the data that the model never saw. The model it
  then saves is retrained on everything, so the saved file has no score of its own.
- **Cross-validation.** Real output of `python compare_models.py --data data/sms.tsv --folds 5`
  (mean ± standard deviation over the folds, for the spam class):

  ```text
  5159 messages (642 spam, 4517 ham) after removing 415 repeated and 0 conflicting messages.
  5-fold stratified cross-validation; scores are for the 'spam' class (mean +/- standard deviation over the folds).
  naive_bayes          precision=1.000+/-0.000  recall=0.508+/-0.062  f1=0.671+/-0.055
  logistic_regression  precision=0.924+/-0.022  recall=0.938+/-0.024  f1=0.930+/-0.007
  linear_svm           precision=0.958+/-0.022  recall=0.928+/-0.035  f1=0.942+/-0.014
  ```

- These numbers were reproduced on Python 3.11, 3.12 and 3.13 with scikit-learn 1.9.1. They describe
  **English SMS spam from this one data set** and nothing else. The data set is old and small, the
  probabilities are not calibrated (the model is trained with balanced class weights), and robustness
  against messages written to fool the model was not tested.
- Splits and folds use a fixed random seed (42), so a rerun on the same file gives the same numbers.
- Data sets that are too small or too lopsided to be split correctly are rejected with an explanation
  instead of producing meaningless scores (every class needs enough messages to appear in the training
  part, the test part and every fold). Small data sets that can be split run with a warning.

## API (development use only)

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/check -H "Content-Type: application/json" \
  -d '{"text":"URGENT! You have won a 1000 prize. Call now to claim your free reward"}'
```

Real responses (HTTP status in brackets):

```text
{"label":"spam","spam_probability":0.992}                       [200]
{"detail":"text: Value error, text must not be blank"}          [422]   body {"text":"   "}
{"detail":"text: Field required"}                               [422]   body {}
{"detail":"text: Input should be a valid string"}               [422]   body {"text":123}
{"detail":"body: JSON decode error"}                            [422]   body that is not JSON
{"detail":"text: String should have at most 10000 characters"}  [422]   10,001 characters
```

- Error messages describe the problem but never contain the submitted text, file paths or stack traces;
  those stay in the server log. An unexpected failure while predicting returns `{"detail":"prediction failed"}` (500).
- The model path comes from the `MODEL_PATH` environment variable (default `model.joblib`). If the model is
  missing, damaged or does not match its checksum, the server **refuses to start**.
- Interactive API documentation is served at `/docs`.
- **This API is for local development.** It has no authentication, no rate limiting and no HTTPS, and it does
  not limit the size of the request body (only the length of the text field, after the body was read). Do not
  expose it to the internet. A public deployment needs, at least, a reverse proxy that provides TLS,
  authentication, rate limits and a body size limit.

## Docker (development image)

```bash
docker build -t phish-guard .
docker run --rm -p 127.0.0.1:8000:8000 phish-guard
```

The build downloads the data (checksum-verified), trains the model and stores it with its checksum, and the
server runs as an unprivileged user. Publish the port on `127.0.0.1` only, as above, because the API has no
authentication. **The image was not built or run while preparing this project** (no Docker daemon was
available); the `docker` job in the CI workflow is what will exercise it on the first push.

## Tests and CI

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover: malformed and tiny data sets, class imbalance, repeated and conflicting messages,
split and fold rules, a model that is missing, damaged, unchecksummed or of the wrong kind, prediction
failures, every API error path (missing, wrongly typed, blank and oversized input), the download script
(checksum mismatch, truncated or oversized downloads, timeouts) and the command line. They use small
generated data, not the real data set, so they run in a few seconds and need no internet (the download
tests talk to a local test server, never to the real source).

`.github/workflows/tests.yml` has two jobs: the tests plus a real `train` and `check` run on the sample
file (Python 3.11 to 3.13), and a Docker job that builds the image and sends one request to it. Workflows
have read-only repository permissions, pin their actions to full commit SHAs, and run for pushes to
`main`, pull requests and manual runs.

## Security notes

- **Never load a model file you did not create.** Models are saved with `joblib`, which uses Python's
  `pickle`. Loading a malicious file runs the attacker's code. The checksum file next to the model only
  detects accidental damage and later changes: whoever can replace the model can also replace the checksum,
  so it is **integrity, not authentication**. `load_model` reads the file once, compares the checksum, and
  only then deserializes the bytes it checked.
- The data download is checked against a pinned SHA-256, has a timeout and a size limit, and is written
  atomically, so a failed or tampered download never replaces a good file.
- The API keeps no copy of the messages and does not log them on purpose (the web server's access log
  records the request line, not the body). Do not send sensitive messages to a server you do not control.
- The direct dependencies are pinned in `requirements.txt` (`==`). Their own dependencies (numpy, scipy,
  starlette, ...) are not, so a fresh install can pick newer versions than the ones tested here.

## Limitations

- Trained on English SMS spam only. No URL or domain analysis, no other languages, no attachments.
- A small, old data set with about 12 % spam (after removing repeats: 642 spam and 4,517 ham messages).
- The 0.5 decision threshold and the probabilities were not tuned or calibrated.
- Not tested against adversarial messages, very long inputs close to the limit under load, or concurrent users.
- The tests print one deprecation warning from the web framework's test client: "Using `httpx` with `starlette.testclient` is
  deprecated; install `httpx2` instead." `requirements-dev.txt` still pins `httpx==0.28.1`, the version the tests were run with.
- Docker, Windows and Python 3.14 were not tested.

## Roadmap

Real work that is still open:

- Decide on a name that matches what the project does (for example `sms-spam-classifier`).
- Test the Docker image, and publish the first CI run.
- Lock the full dependency tree (for example with a lock file) and pin the base image by digest.
- Calibrate the probabilities and choose a threshold on purpose, with a precision/recall trade-off chart.
- Add a second, URL-based feature set or data set before making any claim about phishing.
- Add authentication, rate limiting and a request size limit before any hosting.
- Look at the `httpx2` suggestion in the test warning and update the test dependency if it fits.

## License

Code: MIT, see `LICENSE`. The copyright holder in that file is a placeholder (`YOUR NAME`): replace it
with your own name before publishing. The data set has its own license (CC BY 4.0, see above).
