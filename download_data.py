"""Download the SMS Spam Collection (tab-separated: label<TAB>message) and verify it.

Data set: "SMS Spam Collection" by Tiago A. Almeida and Jose Maria Gomez Hidalgo, UCI Machine
Learning Repository, https://doi.org/10.24432/C5CC84, licensed CC BY 4.0 (credit is required).
The copy fetched here is a public mirror of the same data kept by Kevin Markham:
https://github.com/justmarkham/pycon-2016-tutorial

The URL names one fixed commit of that mirror (not a branch that can move), and the file is accepted
only when its SHA-256 equals EXPECTED_SHA256 as well: a changed, replaced or cut-short download is
rejected instead of being trained on. The target file is written only after that check, in one
atomic step, so a failed run never leaves a broken file behind. The mirror was compared with the
UCI original only by the number of spam and ham messages (4,827 and 747), not message by message.
"""
import argparse
import hashlib
import http.client
import sys
import urllib.request
from pathlib import Path

from phishguard import PhishGuardError, atomic_write

COMMIT = "5339710562cc46c4a32a369d63f1e8751a18a54e"   # the commit of the mirror that holds the file below
URL = f"https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/{COMMIT}/data/sms.tsv"
EXPECTED_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
MAX_BYTES = 5_000_000   # the real file is about 480 KB
TIMEOUT_SECONDS = 30    # applies to each network operation (connect, read), not to the whole download
CREDIT = ("Data: SMS Spam Collection, T. A. Almeida and J. M. Gomez Hidalgo, UCI Machine Learning Repository "
          "(CC BY 4.0), https://doi.org/10.24432/C5CC84")


def sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, timeout=TIMEOUT_SECONDS, max_bytes=MAX_BYTES):
    """Return the body of `url`; refuse anything bigger than `max_bytes`."""
    request = urllib.request.Request(url, headers={"User-Agent": "phish-guard-download/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PhishGuardError(f"The download is larger than {max_bytes} bytes; refusing it")
    return data


def download(dest, url, expected, timeout=TIMEOUT_SECONDS, max_bytes=MAX_BYTES):
    """Fetch `url` into `dest` if `dest` does not already hold the expected data. Returns True if it downloaded."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size <= max_bytes and sha256_of(dest.read_bytes()) == expected:
        return False
    try:
        data = fetch(url, timeout, max_bytes)
    except (OSError, http.client.HTTPException) as exc:  # network errors, HTTP errors (404...), timeouts
        raise PhishGuardError(f"Download failed: {exc}") from exc
    if sha256_of(data) != expected:
        raise PhishGuardError(
            f"The downloaded file has SHA-256 {sha256_of(data)}, not the expected {expected}. "
            "The source has changed or the download was damaged, so nothing was saved"
        )
    atomic_write(dest, data)
    return True


def main(argv=None):
    """Exit codes: 0 = the file is in place and verified, 2 = error (nothing was written)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dest", nargs="?", default="data/sms.tsv", help="where to save the file (default: data/sms.tsv)")
    args = p.parse_args(argv)
    try:
        fetched = download(args.dest, URL, EXPECTED_SHA256)
    except (PhishGuardError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if fetched:
        print(f"Saved {args.dest} (SHA-256 verified)")
    else:
        print(f"{args.dest} already exists and matches the expected SHA-256; nothing to download")
    print(CREDIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
