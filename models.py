#!/usr/bin/env python3
"""models.py — thin clients for the three model families, one call surface each.

Facts-first model split (from the screenshots):
  Gemini (native video, audio-off)  : contact_track, cycle_detect, direction_decide,
                                       rough_segment, edge verifier, bursts
  GPT-5.5                            : per-segment label, template_match
  Claude opus-4-8                    : the single Phase-5 gate (sees 2fps frames)

All clients share a Usage ledger so the orchestrator can print a per-family cost
line. Keys are read from the package dir, falling back to the egoanno root.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ROOTS = [HERE, HERE.parent]                      # package first, then egoanno root

GEMINI_BASE = "https://generativelanguage.googleapis.com"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# model ids (facts-first defaults)
GEMINI_NATIVE = "gemini-3.1-pro-preview"          # native video — EVERY Gemini call uses this
GPT_MODEL = "gpt-5.5"                             # (unused: no GPT stage in facts-first)
CLAUDE_GATE = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")   # Claude stage; env-overridable for A/B
# NOTE: gemini-flash is intentionally absent — facts-first uses pro-preview + opus only.


def read_key(name: str) -> str:
    for root in _ROOTS:
        p = root / name
        if p.exists():
            return p.read_text().strip()
    raise SystemExit(f"models: missing key file {name} (looked in {_ROOTS})")


class Usage:
    """Shared, thread-safe token/call ledger across families."""
    def __init__(self):
        self._lock = threading.Lock()
        self.by = {}

    def add(self, family: str, calls=1, pt=0, ct=0):
        with self._lock:
            d = self.by.setdefault(family, {"calls": 0, "pt": 0, "ct": 0})
            d["calls"] += calls
            d["pt"] += pt
            d["ct"] += ct
        led = os.environ.get("TOKEN_LEDGER")    # cross-process token capture (O_APPEND of a <4KB line is atomic)
        if led:
            try:
                with open(led, "a") as f:
                    f.write(f"{family}\t{calls}\t{pt}\t{ct}\n")
            except Exception:
                pass

    def summary(self) -> str:
        rows = [f"{k}: {v['calls']} calls, {v['pt']}+{v['ct']} tok"
                for k, v in sorted(self.by.items())]
        return " | ".join(rows) if rows else "(no calls)"


USAGE = Usage()


def _strip_fences(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return t.strip()


# --------------------------------------------------------------------------- #
#  GEMINI — native video (File API + generateContent with time offsets)       #
# --------------------------------------------------------------------------- #
class GeminiVideo:
    """Upload an (audio-off, clock-burned) mp4 ONCE, then call generateContent
    with start/end offsets + fps. The whole-clip video stays server-side; each
    stage just names its window."""

    def __init__(self, model: str = GEMINI_NATIVE):
        self.key = read_key(".gemini_key")
        self.model = model
        self.file_uri: str | None = None

    def _post(self, url, data, headers, timeout=1800):
        # patient exponential backoff on 429/5xx (sustained rate-limit windows): up to
        # 9 tries, 4,8,16,32,60,60,60,60s ~= 5min of waiting before giving up.
        last = None
        for attempt in range(9):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read(), r.headers
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 503) and attempt < 8:
                    time.sleep(min(60, 4 * (2 ** attempt)))
                    continue
                raise
        raise last

    def upload(self, path: str | Path) -> str:
        path = Path(path)
        blob = path.read_bytes()
        body, hdrs = self._post(
            f"{GEMINI_BASE}/upload/v1beta/files?key={self.key}",
            json.dumps({"file": {"display_name": path.name}}).encode(),
            {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
             "X-Goog-Upload-Header-Content-Length": str(len(blob)),
             "X-Goog-Upload-Header-Content-Type": "video/mp4",
             "Content-Type": "application/json"})
        body, _ = self._post(hdrs["X-Goog-Upload-URL"], blob,
                             {"X-Goog-Upload-Command": "upload, finalize",
                              "X-Goog-Upload-Offset": "0",
                              "Content-Length": str(len(blob))})
        f = json.loads(body)["file"]
        name, uri = f["name"], f["uri"]
        for _ in range(300):
            if f.get("state") == "ACTIVE":
                self.file_uri = uri
                return uri
            if f.get("state") == "FAILED":
                raise RuntimeError("file processing failed")
            time.sleep(3)
            with urllib.request.urlopen(
                    f"{GEMINI_BASE}/v1beta/{name}?key={self.key}", timeout=60) as r:
                f = json.loads(r.read())
        raise RuntimeError("file never ACTIVE")

    def watch(self, prompt: str, system: str, schema: dict,
              a: float | None = None, b: float | None = None,
              fps: float | None = None, max_tokens: int = 16000,
              temperature: float = 0.15, retries: int = 4,
              media_resolution: str | None = None) -> dict:
        assert self.file_uri, "call upload() first"
        fd = {"file_data": {"mime_type": "video/mp4", "file_uri": self.file_uri}}
        meta = {}
        if a is not None:
            meta["start_offset"] = f"{a:.1f}s"
        if b is not None:
            meta["end_offset"] = f"{b:.1f}s"
        if fps:
            meta["fps"] = min(fps, 24)            # native video_metadata caps at 24fps
        if meta:
            fd["video_metadata"] = meta
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [fd, {"text": prompt}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json",
                                 "responseSchema": schema},
        }
        if media_resolution:                     # e.g. MEDIA_RESOLUTION_LOW -> ~4x more video/window
            body["generationConfig"]["mediaResolution"] = media_resolution
        url = f"{GEMINI_BASE}/v1beta/models/{self.model}:generateContent?key={self.key}"
        last = None
        for attempt in range(retries):
            try:
                raw, _ = self._post(url, json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
                out = json.loads(raw)
                u = out.get("usageMetadata", {})
                USAGE.add("gemini-native", 1, u.get("promptTokenCount", 0),
                          u.get("candidatesTokenCount", 0))
                cand = out["candidates"][0]
                txt = "".join(p.get("text", "")
                              for p in cand.get("content", {}).get("parts", []))
                if not txt:                          # MAX_TOKENS / empty -> grow budget
                    last = f"empty candidate ({cand.get('finishReason')})"
                    body["generationConfig"]["maxOutputTokens"] *= 2
                    time.sleep(2 * (attempt + 1))
                    continue
                return json.loads(txt)
            except json.JSONDecodeError as e:        # truncated -> bigger budget
                last = e
                body["generationConfig"]["maxOutputTokens"] *= 2
                time.sleep(2 * (attempt + 1))
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as e:
                last = getattr(e, "read", lambda: b"")()[:300] if hasattr(e, "read") else e
                time.sleep(4 * (attempt + 1))
        raise RuntimeError(f"gemini.watch failed after {retries}: {last}")


# --------------------------------------------------------------------------- #
#  GEMINI — frame mode (OpenAI-compatible) for bursts                         #
# --------------------------------------------------------------------------- #
class GeminiFrames:
    """Frame-based Gemini (OpenAI-compatible endpoint) for the 30fps micro-bursts.
    Gemini are thinking models: keep a high max_tokens or content comes back empty."""
    BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

    def __init__(self, model: str = GEMINI_NATIVE):
        self.key = read_key(".gemini_key")
        self.model = model

    def call(self, prompt: str, images_b64: list[str], schema: dict | None = None,
             max_tokens: int = 1200, temperature: float = 0.1,
             reasoning: str = "low", retries: int = 3) -> dict | str:
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                   for b in images_b64]
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "temperature": temperature,
                "max_tokens": max(max_tokens, 1500),
                "reasoning_effort": reasoning,
                "messages": [{"role": "user", "content": content}]}
        use_schema = schema is not None
        last = None
        for attempt in range(retries):
            b = dict(body)
            if use_schema:
                b["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": "out", "schema": schema}}
            req = urllib.request.Request(
                f"{self.BASE}/chat/completions", data=json.dumps(b).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
                u = out.get("usage") or {}
                USAGE.add("gemini-burst", 1, u.get("prompt_tokens", 0),
                          u.get("completion_tokens", 0))
                txt = out["choices"][0]["message"]["content"]
                if txt is None:
                    last = "null content"
                    body["max_tokens"] = int(body["max_tokens"] * 2)
                    time.sleep(2 * (attempt + 1))
                    continue
                return json.loads(_strip_fences(txt)) if use_schema else txt
            except urllib.error.HTTPError as e:
                if e.code == 400 and use_schema:
                    use_schema = False
                    continue
                last = f"HTTP {e.code}: {e.read()[:200]}"
                time.sleep(2 * (attempt + 1))
            except (json.JSONDecodeError, urllib.error.URLError, KeyError) as e:
                last = e
                body["max_tokens"] = int(body["max_tokens"] * 2)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"gemini.frames failed after {retries}: {last}")


# --------------------------------------------------------------------------- #
#  GPT-5.5 — labeler + template-match (strict structured outputs)             #
# --------------------------------------------------------------------------- #
def gpt_call(prompt: str, frames_b64: list[str], system: str, schema: dict,
             model: str = GPT_MODEL, max_tokens: int = 8000,
             retries: int = 3) -> dict:
    key = read_key(".openai_key")
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
               for b in frames_b64[:100]]
    content.append({"type": "text", "text": prompt})
    sc = json.loads(json.dumps(schema))              # deep copy
    sc["required"] = list(sc.get("properties", {}).keys())   # strict wants all
    sc["additionalProperties"] = False
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "out", "strict": True, "schema": sc}}}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENAI_URL, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            u = out.get("usage") or {}
            USAGE.add("gpt", 1, u.get("prompt_tokens", 0),
                      u.get("completion_tokens", 0))
            return json.loads(_strip_fences(out["choices"][0]["message"]["content"]))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]}"
            time.sleep(3 * (attempt + 1))
        except (json.JSONDecodeError, urllib.error.URLError, KeyError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gpt_call failed after {retries}: {last}")


# --------------------------------------------------------------------------- #
#  Claude opus-4-8 — the single gate (sees frames)                            #
# --------------------------------------------------------------------------- #
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def claude_call(prompt: str, frames_b64: list[str], system: str, schema: dict,
                model: str = CLAUDE_GATE, max_tokens: int = 12000,
                retries: int = 3) -> dict:
    """Raw HTTP (no SDK needed). Structured output via a forced tool call: the
    model must invoke the 'emit' tool whose input_schema IS our gate schema, so
    the result is schema-valid without prose-JSON parsing."""
    key = read_key(".anthropic_key")
    content = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
               for b in frames_b64[:100]]
    content.append({"type": "text", "text": prompt})
    body = {
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": content}],
        "tools": [{"name": "emit", "description": "Emit the structured gate result.",
                   "input_schema": schema}],
        "tool_choice": {"type": "tool", "name": "emit"},
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ANTHROPIC_URL, data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            u = out.get("usage") or {}
            USAGE.add("claude", 1, u.get("input_tokens", 0), u.get("output_tokens", 0))
            for blk in out.get("content", []):
                if blk.get("type") == "tool_use":
                    return blk["input"]
            # fallback: model returned text instead of a tool call
            txt = "".join(b.get("text", "") for b in out.get("content", [])
                          if b.get("type") == "text")
            return json.loads(_strip_fences(txt))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]}"
            time.sleep(3 * (attempt + 1))
        except (json.JSONDecodeError, urllib.error.URLError, KeyError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"claude_call failed after {retries}: {last}")


# --------------------------------------------------------------------------- #
#  KIMI (Moonshot) — OpenAI-compatible vision chat (for the model A/B)         #
# --------------------------------------------------------------------------- #
KIMI_URL = "https://api.moonshot.ai/v1/chat/completions"
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-latest")


def kimi_call(prompt: str, frames_b64: list[str], system: str, schema: dict,
              max_tokens: int = 8000, retries: int = 3) -> dict:
    """Moonshot Kimi (OpenAI-compatible). json_object mode + the schema's field list described inline."""
    key = read_key(".kimi_key")
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
               for b in frames_b64[:50]]
    content.append({"type": "text", "text": prompt})
    fields = ", ".join((schema.get("properties") or {}).keys())
    sys2 = (system + f"\n\nReturn ONLY a JSON object with exactly these fields: {fields}. "
            "Use English \"yes\"/\"no\" for yes/no fields; object names must be ENGLISH, copied verbatim "
            "from the provided list.")
    no_think = os.environ.get("KIMI_THINKING", "0") != "1"     # default: instant mode (no reasoning)
    body = {"model": KIMI_MODEL, "max_tokens": max(max_tokens, 512),
            "temperature": 0.6 if no_think else 1,             # k2.6: instant requires temp 0.6, thinking requires 1
            "messages": [{"role": "system", "content": sys2}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"}}
    if no_think:
        body["thinking"] = {"type": "disabled"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(KIMI_URL, data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            u = out.get("usage") or {}
            USAGE.add("kimi", 1, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return json.loads(_strip_fences(out["choices"][0]["message"]["content"]))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]}"
            time.sleep(3 * (attempt + 1))
        except (json.JSONDecodeError, urllib.error.URLError, KeyError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"kimi_call failed after {retries}: {last}")


def vlm_call(prompt: str, frames_b64: list[str], system: str, schema: dict, max_tokens: int = 12000) -> dict:
    """Provider dispatch for the dense-grounding A/B. LLM_PROVIDER = claude (default; uses CLAUDE_MODEL) |
    gemini (gemini-3.1-pro-preview via OpenAI-compat) | kimi (Moonshot)."""
    prov = os.environ.get("LLM_PROVIDER", "claude")
    if prov == "gemini":
        gmodel = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")   # pinned 3.1 pro (not flash)
        return GeminiFrames(gmodel).call(system + "\n\n" + prompt, frames_b64, schema,
                                         max_tokens=max(max_tokens, 2000))
    if prov == "kimi":
        return kimi_call(prompt, frames_b64, system, schema, max_tokens=max_tokens)
    return claude_call(prompt, frames_b64, system, schema, model=CLAUDE_GATE, max_tokens=max_tokens)
