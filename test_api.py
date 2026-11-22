import logging

import pytest
from fastapi.testclient import TestClient

import api
from phishguard import MAX_TEXT_CHARS, PhishGuardError, build_model, checksum_path, save_model

JSON = {"Content-Type": "application/json"}


@pytest.fixture
def client(model_file):
    with TestClient(api.create_app(str(model_file))) as test_client:
        yield test_client


def assert_safe_error(res, tmp_path=None):
    """An error response must be short JSON: no stack trace, no file names, no echo of what was sent."""
    body = res.text
    assert res.headers["content-type"].startswith("application/json")
    assert list(res.json()) == ["detail"] and isinstance(res.json()["detail"], str)
    for leak in ("Traceback", ".py", "site-packages", "/home", "/tmp", "joblib", "sklearn"):
        assert leak not in body, leak
    if tmp_path is not None:
        assert str(tmp_path) not in body


# ---------- normal use ----------

def test_check_flags_spam_and_keeps_normal_messages(client):
    spam = client.post("/check", json={"text": "URGENT! You won a FREE prize, click http://bit.ly/x now"})
    assert spam.status_code == 200
    assert set(spam.json()) == {"label", "spam_probability"}
    assert spam.json()["label"] == "spam" and 0.5 <= spam.json()["spam_probability"] <= 1.0
    ham = client.post("/check", json={"text": "Can we move lunch to tomorrow, after the meeting?"})
    assert ham.status_code == 200 and ham.json()["label"] == "ham" and ham.json()["spam_probability"] < 0.5


def test_probability_is_rounded_to_three_decimals(client):
    probability = client.post("/check", json={"text": "free prize now"}).json()["spam_probability"]
    assert probability == round(probability, 3)


def test_text_is_limited_to_the_maximum_length(client):
    assert client.post("/check", json={"text": "x" * MAX_TEXT_CHARS}).status_code == 200
    res = client.post("/check", json={"text": "x" * (MAX_TEXT_CHARS + 1)})
    assert res.status_code == 422 and "at most" in res.json()["detail"]
    assert "xxxxxxxx" not in res.text                                           # the text is not echoed back


def test_unknown_extra_fields_are_ignored(client):
    assert client.post("/check", json={"text": "hello there", "extra": 1}).status_code == 200


def test_the_model_is_loaded_once_at_startup_not_per_request(model_file, monkeypatch):
    calls = []
    real = api.load_model
    monkeypatch.setattr(api, "load_model", lambda path: calls.append(path) or real(path))
    with TestClient(api.create_app(str(model_file))) as client:
        for _ in range(3):
            assert client.post("/check", json={"text": "hello there"}).status_code == 200
    assert calls == [str(model_file)]


def test_the_model_path_can_come_from_the_environment(model_file, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    with TestClient(api.create_app()) as client:
        assert client.post("/check", json={"text": "free prize"}).status_code == 200


# ---------- bad requests ----------

@pytest.mark.parametrize("body, field_message", [
    ({}, "text: Field required"),
    ({"txt": "typo in the field name"}, "text: Field required"),
    ({"text": ""}, "text: String should have at least 1 character"),
    ({"text": "   "}, "text: Value error, text must not be blank"),
    ({"text": "\n\t "}, "text: Value error, text must not be blank"),
    ({"text": None}, "text: Input should be a valid string"),
    ({"text": 123}, "text: Input should be a valid string"),
    ({"text": 1.5}, "text: Input should be a valid string"),
    ({"text": True}, "text: Input should be a valid string"),
    ({"text": ["free", "prize"]}, "text: Input should be a valid string"),
    ({"text": {"a": 1}}, "text: Input should be a valid string"),
])
def test_wrong_or_missing_fields_are_rejected_with_a_short_reason(client, tmp_path, body, field_message):
    res = client.post("/check", json=body)
    assert res.status_code == 422 and res.json() == {"detail": field_message}
    assert_safe_error(res, tmp_path)


@pytest.mark.parametrize("raw", ["[]", '"just a string"', "123", "null", "true"])
def test_a_body_that_is_not_a_json_object_is_rejected(client, raw):
    res = client.post("/check", content=raw, headers=JSON)
    assert res.status_code == 422 and res.json()["detail"].startswith("body:")
    assert_safe_error(res)


@pytest.mark.parametrize("raw", ["", "{not json", '{"text": "unterminated', "\x00\x01"])
def test_broken_json_is_rejected(client, raw):
    res = client.post("/check", content=raw, headers=JSON)
    assert res.status_code == 422
    assert_safe_error(res)


def test_a_non_json_content_type_is_rejected(client):
    res = client.post("/check", content='{"text": "hello"}', headers={"Content-Type": "text/plain"})
    assert res.status_code == 422
    assert_safe_error(res)
    assert client.post("/check", data={"text": "hello"}).status_code == 422       # an HTML-form style body


def test_wrong_method_and_unknown_route_give_json_errors(client):
    assert client.get("/check").status_code == 405
    assert client.get("/nothing-here").status_code == 404
    assert_safe_error(client.get("/check"))


# ---------- a model that cannot be used ----------

def test_the_server_refuses_to_start_without_a_model(tmp_path):
    with pytest.raises(PhishGuardError, match="Cannot read model file"):
        with TestClient(api.create_app(str(tmp_path / "missing.joblib"))):
            pass


def test_the_server_refuses_to_start_with_a_model_that_has_no_matching_checksum(model_file):
    checksum_path(model_file).unlink()
    with pytest.raises(PhishGuardError, match="checksum"):
        with TestClient(api.create_app(str(model_file))):
            pass


def test_the_server_refuses_to_start_with_a_model_that_cannot_predict(tmp_path):
    path = tmp_path / "model.joblib"
    save_model(build_model(), path)                                                # never fitted: loads, but cannot predict
    with pytest.raises(PhishGuardError, match="not a phish-guard model"):
        with TestClient(api.create_app(str(path))):
            pass


def test_startup_problems_are_logged_for_the_operator(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="phishguard.api"):
        with pytest.raises(PhishGuardError):
            with TestClient(api.create_app(str(tmp_path / "missing.joblib"))):
                pass
    assert "cannot start" in caplog.text


class ExplodingModel:
    classes_ = ["ham", "spam"]

    def predict_proba(self, texts):
        raise RuntimeError("secret detail: /srv/private/model.joblib is damaged")


class NanModel:
    classes_ = ["ham", "spam"]

    def predict_proba(self, texts):
        return [[float("nan"), float("nan")]]


@pytest.mark.parametrize("broken", [ExplodingModel(), NanModel()])
def test_prediction_failures_are_a_generic_500_without_details(client, tmp_path, caplog, broken):
    client.app.state.model = broken
    with caplog.at_level(logging.ERROR, logger="phishguard.api"):
        res = client.post("/check", json={"text": "free prize"})
    assert res.status_code == 500 and res.json() == {"detail": "prediction failed"}
    assert_safe_error(res, tmp_path)
    assert "/srv/private" not in res.text and "secret detail" not in res.text
    assert "prediction failed" in caplog.text                                       # the details go to the server log only
