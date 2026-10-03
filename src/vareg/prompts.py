"""Prompt versioning: one file per version plus a manifest.

The front matter is a deliberately trivial ``key: value`` block between two
``---`` lines — parsed in a dozen lines of stdlib rather than claiming YAML
support this repo does not carry. Two of those keys are load-bearing:

``tools``
    The capability surface handed to the backend. A tool missing from this list
    is not exposed to the model, so adding ``cancel_booking`` in v2 is a real
    behaviour change (see scenarios/cancellation_policy.json), not a comment.
``clarify_on_missing_slots``
    Whether a missing slot produces a question or a guessed call.

The manifest carries what a reviewer needs to audit a bump: version, date and a
change note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptError(ValueError):
    """Bad manifest, missing version file, or malformed front matter."""


@dataclass(frozen=True)
class PromptSpec:
    version: str
    date: str
    change: str
    body: str
    tools: tuple[str, ...]
    clarify_on_missing_slots: bool
    faqs: tuple[tuple[tuple[str, ...], str], ...]

    def faq_answer(self, text: str) -> str | None:
        """First FAQ whose trigger words appear in *text*, else None."""
        haystack = text.casefold()
        for triggers, answer in self.faqs:
            if any(trigger in haystack for trigger in triggers):
                return answer
        return None


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PromptError("prompt file must start with a '---' front matter block")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise PromptError("front matter block is not closed by a second '---'") from exc
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise PromptError(f"front matter line is not 'key: value': {line!r}")
        meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip("\n")
    return meta, body


def parse_faqs(body: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    """``FAQ: trigger|trigger => answer text`` lines inside the prompt body."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("FAQ:"):
            continue
        payload = line[len("FAQ:") :].strip()
        if "=>" not in payload:
            raise PromptError(f"FAQ line has no '=>': {line!r}")
        triggers, answer = payload.split("=>", 1)
        words = tuple(w.strip().casefold() for w in triggers.split("|") if w.strip())
        if not words or not answer.strip():
            raise PromptError(f"FAQ line needs a trigger and an answer: {line!r}")
        out.append((words, answer.strip()))
    return tuple(out)


def load_manifest(prompt_dir: Path = PROMPT_DIR) -> dict:
    path = prompt_dir / "manifest.json"
    if not path.is_file():
        raise PromptError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data.get("versions"), list) or not data["versions"]:
        raise PromptError("manifest must contain a non-empty 'versions' list")
    if "default" not in data:
        raise PromptError("manifest must declare a 'default' version")
    return data


def load_spec(version: str | None = None, prompt_dir: Path = PROMPT_DIR) -> PromptSpec:
    manifest = load_manifest(prompt_dir)
    wanted = version or manifest["default"]
    notes = {entry.get("version"): entry for entry in manifest["versions"]}
    if wanted not in notes:
        raise PromptError(
            f"unknown prompt version {wanted!r}; manifest knows: "
            + ", ".join(sorted(str(v) for v in notes))
        )
    path = prompt_dir / f"{wanted}.md"
    if not path.is_file():
        raise PromptError(f"prompt file not found: {path}")
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    if meta.get("version") != wanted:
        raise PromptError(
            f"{path.name} declares version {meta.get('version')!r} but the manifest calls it {wanted!r}"
        )
    tools = tuple(t.strip() for t in meta.get("tools", "").split(",") if t.strip())
    if not tools:
        raise PromptError(f"{path.name} declares no tools")
    entry = notes[wanted]
    return PromptSpec(
        version=wanted,
        date=str(entry.get("date", meta.get("date", ""))),
        change=str(entry.get("change", "")),
        body=body,
        tools=tools,
        clarify_on_missing_slots=meta.get("clarify_on_missing_slots", "").lower() == "true",
        faqs=parse_faqs(body),
    )
