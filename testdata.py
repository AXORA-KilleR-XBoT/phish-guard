"""Small synthetic datasets shared by the tests (no real messages, nothing to download)."""


def rows(n_spam, n_ham):
    """(label, text) pairs; every text is different, so none of them counts as a repeat."""
    spam = [("spam", f"WINNER! Claim your free prize number {i} now, click the link") for i in range(n_spam)]
    ham = [("ham", f"See you at lunch about project {i}, are we still meeting tomorrow?") for i in range(n_ham)]
    return spam + ham


def write_tsv(path, pairs):
    path.write_text("".join(f"{label}\t{text}\n" for label, text in pairs), encoding="utf-8")
    return path
