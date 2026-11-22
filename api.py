"""HTTP API for phish-guard: POST /check {"text": "..."} -> label + spam probability.

FOR LOCAL DEVELOPMENT ONLY. This API has no authentication, no rate limiting, no HTTPS and no
limit on the size of the request body. Do not expose it to the internet as it is.

Run it with:  uvicorn api:app --host 127.0.0.1 --port 8000
The model file is read from the MODEL_PATH environment variable (default: model.joblib). If the
model cannot be loaded (missing, no matching checksum file, damaged), the server refuses to start.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from phishguard import MAX_TEXT_CHARS, load_model, predict

log = logging.getLogger("phishguard.api")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def create_app(model_path=None):
    """Build the app. `model_path` overrides the MODEL_PATH environment variable (used by the tests)."""

    @asynccontextmanager
    async def lifespan(app):
        path = model_path or os.environ.get("MODEL_PATH", "model.joblib")
        try:
            app.state.model = load_model(path)
        except Exception as exc:
            log.error("cannot start: %s", exc)  # the operator sees the details; clients never do
            raise
        yield

    app = FastAPI(title="phish-guard API (development use only)", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Say what is wrong with the request, but never echo the submitted text back.
        problems = []
        for error in exc.errors():
            fields = [str(part) for part in error["loc"][1:] if isinstance(part, str)]
            problems.append(f"{'.'.join(fields) or 'body'}: {error['msg']}")
        return JSONResponse(status_code=422, content={"detail": "; ".join(problems)})

    @app.post("/check")
    def check(message: Message, request: Request):
        try:
            label, prob = predict(request.app.state.model, message.text)
        except Exception:
            log.exception("prediction failed")  # the stack trace stays in the server log
            return JSONResponse(status_code=500, content={"detail": "prediction failed"})
        return {"label": label, "spam_probability": round(prob, 3)}

    return app


app = create_app()
