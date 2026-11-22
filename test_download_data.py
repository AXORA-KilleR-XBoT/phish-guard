import hashlib
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import download_data
from download_data import download, fetch, main
from phishguard import PhishGuardError

GOOD = b"ham\thello there\nspam\tfree prize, click now\n"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()


class Handler(BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):
        Handler.hits.append(self.path)
        if self.path == "/good":
            body = GOOD
        elif self.path == "/other":
            body = b"ham\tsomething else\n"
        elif self.path == "/big":
            body = b"x" * 10_000
        elif self.path == "/slow":
            time.sleep(2)
            body = GOOD
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:  # the test gave up waiting
            pass

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def server(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")            # a local test server must not go through a proxy
    monkeypatch.setenv("NO_PROXY", "*")
    Handler.hits = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def folder_contents(path):
    return sorted(p.name for p in path.iterdir())


def test_a_good_download_is_saved_exactly(server, tmp_path):
    dest = tmp_path / "data" / "sms.tsv"
    assert download(dest, server + "/good", GOOD_SHA) is True
    assert dest.read_bytes() == GOOD
    assert folder_contents(dest.parent) == ["sms.tsv"]                   # no temporary file left behind


def test_nothing_is_downloaded_when_the_file_is_already_correct(server, tmp_path):
    dest = tmp_path / "sms.tsv"
    dest.write_bytes(GOOD)
    assert download(dest, server + "/good", GOOD_SHA) is False
    assert Handler.hits == []


def test_a_file_with_other_content_is_replaced(server, tmp_path):
    dest = tmp_path / "sms.tsv"
    dest.write_bytes(b"ham\told local copy\n")
    assert download(dest, server + "/good", GOOD_SHA) is True
    assert dest.read_bytes() == GOOD


def test_content_with_the_wrong_checksum_is_rejected_and_not_saved(server, tmp_path):
    dest = tmp_path / "sms.tsv"
    with pytest.raises(PhishGuardError, match="not the expected"):
        download(dest, server + "/other", GOOD_SHA)                      # the server now serves something different
    assert folder_contents(tmp_path) == []


def test_a_failed_download_does_not_destroy_the_existing_file(server, tmp_path):
    dest = tmp_path / "sms.tsv"
    dest.write_bytes(b"ham\told local copy\n")
    with pytest.raises(PhishGuardError):
        download(dest, server + "/other", GOOD_SHA)
    assert dest.read_bytes() == b"ham\told local copy\n"
    assert folder_contents(tmp_path) == ["sms.tsv"]


def test_http_errors_are_clean_errors(server, tmp_path):
    with pytest.raises(PhishGuardError, match="Download failed: HTTP Error 404"):
        download(tmp_path / "sms.tsv", server + "/missing", GOOD_SHA)
    assert folder_contents(tmp_path) == []


def test_a_response_bigger_than_the_limit_is_refused(server, tmp_path):
    with pytest.raises(PhishGuardError, match="larger than 1000 bytes"):
        download(tmp_path / "sms.tsv", server + "/big", GOOD_SHA, max_bytes=1000)
    assert folder_contents(tmp_path) == []


def test_a_server_that_does_not_answer_in_time_is_a_clean_error(server, tmp_path):
    with pytest.raises(PhishGuardError, match="Download failed"):
        download(tmp_path / "sms.tsv", server + "/slow", GOOD_SHA, timeout=0.3)
    assert folder_contents(tmp_path) == []


def test_a_closed_port_is_a_clean_error(tmp_path):
    with pytest.raises(PhishGuardError, match="Download failed"):
        download(tmp_path / "sms.tsv", "http://127.0.0.1:1/sms.tsv", GOOD_SHA, timeout=2)
    assert folder_contents(tmp_path) == []


def test_fetch_returns_the_body(server):
    assert fetch(server + "/good") == GOOD


def test_the_pinned_checksum_is_a_well_formed_sha256():
    assert len(download_data.EXPECTED_SHA256) == 64 and set(download_data.EXPECTED_SHA256) <= set("0123456789abcdef")
    assert download_data.URL.startswith("https://")                       # the real source is only ever fetched over HTTPS
    assert re.fullmatch(r"[0-9a-f]{40}", download_data.COMMIT)           # a fixed commit, not a branch that can move
    assert f"/{download_data.COMMIT}/" in download_data.URL


# ---------- command line ----------

def test_cli_success_and_repeat_run(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download_data, "URL", server + "/good")
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", GOOD_SHA)
    dest = tmp_path / "d" / "sms.tsv"
    assert main([str(dest)]) == 0
    out = capsys.readouterr().out
    assert f"Saved {dest}" in out and "CC BY 4.0" in out and "doi.org/10.24432/C5CC84" in out
    assert main([str(dest)]) == 0
    assert "nothing to download" in capsys.readouterr().out
    assert Handler.hits == ["/good"]                                      # the second run did not touch the network


def test_cli_failure_exits_with_2_and_writes_nothing(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download_data, "URL", server + "/other")
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", GOOD_SHA)
    dest = tmp_path / "sms.tsv"
    assert main([str(dest)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err
    assert not dest.exists()


def test_the_body_is_read_with_a_size_limit(monkeypatch):
    """The reader must never be asked for more than max_bytes + 1 bytes, however large the server's answer is."""
    asked = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            asked.append(size)
            return b"x" * (size if size >= 0 else 10**9)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", lambda request, timeout: Response())
    with pytest.raises(PhishGuardError, match="larger than"):
        fetch("http://example.invalid/", max_bytes=100)
    assert asked == [101]
