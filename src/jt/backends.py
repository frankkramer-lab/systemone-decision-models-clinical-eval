"""Backends that turn a decision request into a probability per option (in the question's option order).

- JeffLocal: a Jeff checkpoint, or an untrained base model with Jeff's readout initialised from the LM-head rows of the
  answer codes (temperature 1). Both use Jeff's exact prompt (jeff.model.decision_messages).
- ChatLogprobs: any OpenAI-compatible chat endpoint (Open WebUI -> vLLM). Same prompt; one generated token; the top
  log-probabilities of the answer-code tokens are renormalised over the options. The mass outside the options is kept.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx

from jt import config as C

CODES = [chr(ord("A") + i) for i in range(26)]  # Jeff's first 26 codes; every task here has at most 5 options


class Backend:
    local = True
    host = "localhost"
    model = ""

    def check_tier(self, tier):
        if tier == "open":
            return
        if self.host not in C.CREDENTIALED_HOSTS or self.model.startswith(C.DENY_PREFIXES):
            raise PermissionError(f"{self.model}@{self.host} may not receive credentialed data "
                                  f"(allowed hosts {sorted(C.CREDENTIALED_HOSTS)}, denied prefixes {C.DENY_PREFIXES})")


class JeffLocal(Backend):
    def __init__(self, name, checkpoint=None, base_model=None, revision=None, batch_size=8):
        from jeff.model import DecisionModel
        self.model = name
        kw = {"checkpoint": checkpoint} if checkpoint else {"base_model": base_model, "revision": revision}
        self.net = DecisionModel(**kw)
        self.batch_size = batch_size

    def predict(self, rows):
        t = time.perf_counter()
        probs = self.net.predict(rows, batch_size=self.batch_size)
        per = (time.perf_counter() - t) / len(rows)
        return [{"probs": p, "seconds": per, "outside": 0.0} for p in probs]


class ChatLogprobs(Backend):
    def __init__(self, model, base_url=None, api_key=None, extra=None, concurrency=8, timeout=300, skip_think=False):
        self.model = model
        self.skip_think = skip_think  # pre-fill an empty "<think></think>" so the first token is the answer (GLM-5.3-Flash)
        self.base_url = (base_url or C.ENV["LLM_BASE_URL"]).rstrip("/")
        self.host = urlparse(self.base_url).hostname
        self.headers = {"Authorization": f"Bearer {api_key or C.ENV['LLM_API_KEY']}"}
        self.extra = extra if extra is not None else {"chat_template_kwargs": {"enable_thinking": False}}
        self.concurrency = concurrency
        self.client = httpx.Client(timeout=timeout)

    def one(self, row):
        from jeff.model import decision_messages, options
        n = len(options(row["question"])[0])
        messages = decision_messages(row, CODES)
        prefill = {"messages": messages + [{"role": "assistant", "content": "<think></think>"}],
                   "continue_final_message": True, "add_generation_prompt": False} if self.skip_think else {}
        body = {"model": self.model, "messages": messages, "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 20, "stream": False, **self.extra, **prefill}
        attempt, waited = 0, 0.0
        while True:
            t = time.perf_counter()
            r = None
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=body)
                if r.status_code == 429 and waited < 3600:  # rate limit: wait as told, without using up the retries
                    pause = float(r.headers.get("retry-after", 10))
                    time.sleep(pause)
                    waited += pause
                    continue
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                attempt += 1
        seconds = time.perf_counter() - t
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        mass = {}
        for entry in top:  # " A" and "A" are both the answer A
            token = entry["token"].strip()
            if token in CODES[:n]:
                mass[token] = mass.get(token, 0.0) + math.exp(entry["logprob"])
        total = sum(mass.values())
        if total == 0:  # no option among the top 20: uniform, flagged by outside = 1
            return {"probs": [1 / n] * n, "seconds": seconds, "outside": 1.0}
        return {"probs": [mass.get(c, 0.0) / total for c in CODES[:n]], "seconds": seconds, "outside": 1 - total}

    def predict(self, rows):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self.one, rows))


class JevHTTP(Backend):
    """The Jev request format (POST {state, questions} -> answers): TypeSafe's hosted Jev, or OpenJev's helper (shim.py).
    One question per request, so each decision is timed alone. A noul answer is P(true); choice answers carry the
    full distribution (rounded to 4 decimals by both servers)."""

    def __init__(self, name, url, api_key=None, model=None, concurrency=8, timeout=120):
        self.model = name
        self.url = url
        self.host = urlparse(url).hostname
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model_id = model
        self.concurrency = concurrency
        self.client = httpx.Client(timeout=timeout)

    def one(self, row):
        question = {k: v for k, v in row["question"].items() if k != "true_first"}  # not part of the Jev format
        body = {"state": row["state"], "questions": {"q": question}, **({"model": self.model_id} if self.model_id else {})}
        for attempt in range(6):
            t = time.perf_counter()
            try:
                r = self.client.post(self.url, headers=self.headers, json=body)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        seconds = time.perf_counter() - t
        a = r.json()["answers"]["q"]
        if row["question"]["type"] == "noul":
            p = float(a["noul"])
            probs = [p, 1 - p] if row["question"].get("true_first") else [1 - p, p]
        else:
            probs = [float(a["probabilities"][k]) for k in row["question"]["criteria"]]
            total = sum(probs)
            probs = [v / total for v in probs]
        return {"probs": probs, "seconds": seconds, "outside": 0.0}

    def predict(self, rows):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self.one, rows))


def make(name):
    """Backends by name; see README for the list."""
    jeff = {"jeff-0.8b": "jeff-0.8b-v1.2", "jeff-2b": "jeff-2b-v1.2"}
    bases = {"qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
             "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")}
    if name in jeff:
        return JeffLocal(name, checkpoint=str(C.MODELS / jeff[name]))
    if name in bases:
        return JeffLocal(name, base_model=bases[name][0], revision=bases[name][1])
    if name.startswith("chat:"):  # chat:<model id on the endpoint>
        if name == "chat:GLM-5.3-Flash":  # institutional server; would open with "</think>" unless pre-filled
            return ChatLogprobs("GLM-5.3-Flash", skip_think=True, concurrency=int(os.environ.get("JT_CONCURRENCY", "2")))
        local = os.environ.get("JT_BASE_URL")  # e.g. a vLLM on loopback for the latency runs; no key is sent there
        return ChatLogprobs(name.removeprefix("chat:"), base_url=local, api_key="none" if local else None,
                            concurrency=int(os.environ.get("JT_CONCURRENCY", "8")))
    if name.startswith("or:"):  # or:<OpenRouter model id>; open tasks only (check_tier refuses openrouter.ai)
        return ChatLogprobs(name.removeprefix("or:"), base_url="https://openrouter.ai/api/v1",
                            api_key=C.ENV["OPENROUTER_API_KEY"],
                            extra={"reasoning": {"enabled": False},  # first-token readout, as for the other LLM arms
                                   "provider": {"require_parameters": True, "data_collection": "deny"}},
                            concurrency=int(os.environ.get("JT_CONCURRENCY", "8")))
    if name.startswith("reason:"):  # reason:<served model id>, local vLLM only (scripts/serve_qwen38.sh)
        return ChatReasoning(name.removeprefix("reason:"), os.environ.get("JT_BASE_URL", "http://127.0.0.1:8001/v1"),
                             os.environ.get("JT_TOKENIZER", str(C.MODELS / "qwen3.8-27b-fp8")),
                             max_think=int(os.environ.get("JT_MAX_THINK", "8192")),
                             effort=os.environ.get("JT_EFFORT", "xhigh"),
                             concurrency=int(os.environ.get("JT_CONCURRENCY", "32")))
    if name == "openjev-fp8":  # helper on loopback in front of a local vLLM (scripts/serve_openjev.sh)
        return JevHTTP(name, "http://127.0.0.1:3000/v1/systemone")
    if name == "jev":  # hosted; open tier only (check_tier refuses credentialed data for this host)
        return JevHTTP(name, "https://api.typesafe.ai/v1/systemone", api_key=C.ENV["TYPESAFE_API_KEY"], model="jev-1.13.0")
    raise KeyError(name)


class ChatReasoning(Backend):
    """Generative use with reasoning, on a local vLLM (completions API, the model's own chat template, thinking on).

    Step 1 samples the reasoning until "</think>" with the model card's thinking-mode settings (temperature 1.0,
    top_p 0.95, top_k 20; seed fixed per prompt), capped at max_think tokens (then closed and flagged). Step 2 appends
    "</think>" and reads the option-code probabilities at the first answer token, renormalised as in ChatLogprobs.
    Same prompt (decision_messages) as every other arm."""

    def __init__(self, model, base_url, tokenizer, max_think=8192, effort="xhigh", concurrency=32, timeout=3600):
        from transformers import AutoTokenizer
        self.model = model
        self.effort = effort  # the template's default; xhigh / medium / low
        self.base_url = base_url.rstrip("/")
        self.host = urlparse(self.base_url).hostname
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.max_think = max_think
        self.concurrency = concurrency
        self.client = httpx.Client(timeout=timeout)

    def post(self, body):
        for attempt in range(6):
            try:
                r = self.client.post(f"{self.base_url}/completions", json={"model": self.model, **body})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                return r.json()["choices"][0]
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def one(self, row):
        import zlib
        from jeff.model import decision_messages, options
        n = len(options(row["question"])[0])
        prompt = self.tok.apply_chat_template(decision_messages(row, CODES), tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True, reasoning_effort=self.effort)
        t = time.perf_counter()
        think = self.post({"prompt": prompt, "max_tokens": self.max_think, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                           "seed": zlib.crc32(prompt.encode()), "stop": ["</think>"]})
        reasoning = think["text"]
        truncated = think["finish_reason"] == "length"
        answer = self.post({"prompt": prompt + reasoning.rstrip() + "\n</think>\n\n", "max_tokens": 1, "temperature": 0,
                            "logprobs": 20})
        seconds = time.perf_counter() - t
        mass = {}
        for token, lp in answer["logprobs"]["top_logprobs"][0].items():
            token = token.strip()
            if token in CODES[:n]:
                mass[token] = mass.get(token, 0.0) + math.exp(lp)
        total = sum(mass.values())
        extra = {"think_tokens": len(self.tok(reasoning)["input_ids"]), "truncated": truncated, "reasoning": reasoning}
        if total == 0:
            return {"probs": [1 / n] * n, "seconds": seconds, "outside": 1.0, "extra": extra}
        return {"probs": [mass.get(c, 0.0) / total for c in CODES[:n]], "seconds": seconds, "outside": 1 - total,
                "extra": extra}

    def predict(self, rows):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self.one, rows))
