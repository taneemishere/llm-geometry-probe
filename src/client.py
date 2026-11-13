"""NVIDIA NIM client with on-disk response cache and token/budget logging."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CACHE_DIR = Path("results/cache")
USAGE_LOG = Path("results/usage.jsonl")

# Conservative assumed prices (USD per 1M tokens) recorded per prereg; used only
# for the $10 stop rule, never for claims.
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 9.0
BUDGET_USD = 10.0


def _key(model: str, prompt: str, sample_idx: int) -> str:
    return hashlib.sha256(f"{model}|{prompt}|{sample_idx}".encode()).hexdigest()


def estimated_cost(log_path=None) -> float:
    log = Path(log_path) if log_path else USAGE_LOG
    if not log.exists():
        return 0.0
    tin = tout = 0
    for ln in log.read_text().splitlines():
        try:
            u = json.loads(ln)
            tin += u.get("prompt_tokens", 0)
            tout += u.get("completion_tokens", 0)
        except json.JSONDecodeError:
            continue
    return tin / 1e6 * PRICE_IN_PER_MTOK + tout / 1e6 * PRICE_OUT_PER_MTOK


class NIMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 8192,
                 temperature: float = 1.0, top_p: float = 1.0, extra: dict | None = None,
                 cache_dir=CACHE_DIR, usage_log=USAGE_LOG):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.extra = extra or {}
        self.cache_dir = Path(cache_dir)
        self.usage_log = Path(usage_log)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat_messages(self, messages: list, key_parts: str) -> dict:
        """Multi-turn call; cache key = sha256(model | messages | key_parts)."""
        key = hashlib.sha256(
            f"{self.model}|{json.dumps(messages, sort_keys=True)}|{key_parts}".encode()
        ).hexdigest()
        return self._call(messages, key, key_parts)

    def chat(self, prompt: str, sample_idx: int) -> dict:
        """Return {'content': str, 'usage': dict, 'cached': bool}."""
        key = _key(self.model, prompt, sample_idx)
        return self._call([{"role": "user", "content": prompt}], key, sample_idx)

    def _call(self, messages: list, key: str, tag) -> dict:
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            d = json.loads(cpath.read_text())
            d["cached"] = True
            return d
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
            **self.extra,
        }
        last_err = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(
                    API_URL,
                    data=json.dumps(body).encode(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                last_err = e
                code = e.code
                if code in (400, 401, 403, 404, 422):
                    raise
                time.sleep(min(2 ** attempt * 10, 180))
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt * 10, 180))
        else:
            raise RuntimeError(f"API failed after retries: {last_err}")
        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        out = {"content": choice["message"].get("content") or "", "usage": usage, "cached": False}
        rec = {"model": self.model, "key": key, "tag": str(tag),
               "prompt_tokens": usage.get("prompt_tokens", 0),
               "completion_tokens": usage.get("completion_tokens", 0)}
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        cpath.write_text(json.dumps(out))
        return out
