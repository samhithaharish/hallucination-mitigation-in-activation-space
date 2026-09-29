"""Strip outputs and scan for secrets before a notebook goes into git.

    python scripts/clean_notebook.py path/to/notebook.ipynb            # writes <name>.clean.ipynb
    python scripts/clean_notebook.py path/to/notebook.ipynb --check    # scan only, no write

Removes cell outputs, execution counts and Colab run metadata (which includes your name
and user id), and exits non-zero if it finds anything that looks like a credential
(Hugging Face tokens, OpenAI/Anthropic-style keys, AWS keys) in the source cells.
Findings are reported by cell number and pattern, never by printing the secret.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "huggingface token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "openai-style key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "anthropic-style key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic token assignment": re.compile(
        r"""(?i)\b(token|api[_-]?key|secret|password)\b\s*=\s*["'][^"']{12,}["']"""),
}


def scan(nb: dict) -> list[tuple[int, str]]:
    findings = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(src):
                findings.append((i, name))
    return findings


def clean(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata", {})
        for k in ("executionInfo", "outputId", "colab"):
            meta.pop(k, None)
    nb.get("metadata", {}).pop("colab", None)
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = Path(args.notebook)
    nb = json.loads(path.read_text())
    findings = scan(nb)
    for cell_no, name in findings:
        print(f"[SECRET?] cell {cell_no}: looks like a {name}", file=sys.stderr)

    if not args.check:
        out = path.with_suffix(".clean.ipynb")
        out.write_text(json.dumps(clean(nb), indent=1))
        print(f"wrote {out}")

    if findings:
        print("Revoke any real credentials, replace them with os.environ lookups, "
              "then re-run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
