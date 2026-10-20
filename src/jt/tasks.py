"""Tasks as Jev/Jeff decision requests: every item becomes one or more questions with named options.

All models (Jeff, untrained bases, LLMs, Jev) receive the same questions. Option texts are fixed here before any test
run and are not tuned on test data. Few-shot: one labelled example per option, drawn once per (task, question, seed)
from the training pool and shown to every model in the same place (the state, before the text to classify).
"""
import json
import random
from dataclasses import dataclass

from jt import config as C

CXR_FINDINGS = ["Pleural Effusion", "Edema", "Pneumonia", "Cardiomegaly"]
# Value definitions from a previously frozen annotation guideline (v1), with the finding name filled in.
CXR_VALUES = {
    "positive": "The text states that {f} is present.",
    "negative": "The text states that {f} is absent.",
    "uncertain": "The text refers to {f} but leaves open whether it is present.",
    "not_mentioned": "The text does not refer to {f} at all.",
}
MEDABS_CLASSES = ["neoplasms", "digestive system diseases", "nervous system diseases", "cardiovascular diseases",
                  "general pathological conditions"]
HALLMARKS = [
    "sustaining proliferative signaling", "evading growth suppressors", "resisting cell death",
    "enabling replicative immortality", "inducing angiogenesis", "activating invasion and metastasis",
    "genomic instability and mutation", "tumor promoting inflammation", "cellular energetics",
    "avoiding immune destruction",
]


@dataclass(frozen=True)
class Task:
    name: str
    tier: str  # "open" (any backend) or "credentialed" (local or allowed hosts only)
    questions: dict  # question id -> Jev question (without few-shot)
    test: str
    train: str

    def gold(self, item, qid):
        """Gold answer key for one question: an option key for choice, "true"/"false" for noul."""
        if self.name == "medabs":
            return key(item["label"])
        if self.name == "hoc":
            return "true" if item["labels"][qid] else "false"
        return item["labels"][qid]


def key(name):
    return name.replace(" ", "_")


TASKS = {
    "medabs": Task("medabs", "open", {"condition": {
        "type": "choice", "instructions": "Which disease class does this medical abstract describe?",
        "criteria": {key(c): c for c in MEDABS_CLASSES}}}, "medabs_test", "medabs_train"),
    "hoc": Task("hoc", "open", {h: {
        "type": "noul", "instructions": f"Does this cancer research abstract describe the cancer hallmark '{h}'?",
        "criteria": {"true": f"The abstract describes {h}.", "false": f"The abstract does not describe {h}."}}
        for h in HALLMARKS}, "hoc_test", "hoc_train"),
    "cxr": Task("cxr", "credentialed", {f: {
        "type": "choice", "instructions": f"How does this chest X-ray report text describe {f.lower()}?",
        "criteria": {v: d.format(f=f.lower()) for v, d in CXR_VALUES.items()}} for f in CXR_FINDINGS},
        "cxr_test", "cxr_train"),
}


def load(name):
    with open(C.DATA_PROCESSED / f"{name}.jsonl") as fp:
        return [json.loads(line) for line in fp]


def option_keys(question):
    return list(question["criteria"]) if question["type"] == "choice" else ["false", "true"]


def few_shot(task, qid, seed):
    """One training example per option of this question, in a seeded random order."""
    pool = load(task.train)
    rng = random.Random(f"{task.name}-{qid}-{seed}")
    rng.shuffle(pool)
    picked = []
    for option in option_keys(task.questions[qid]):
        picked.append(next(item for item in pool if task.gold(item, qid) == option))
    rng.shuffle(picked)
    return [(item["text"], task.gold(item, qid)) for item in picked]


def state(text, examples):
    if not examples:
        return text
    words = {"true": "yes", "false": "no"}  # noul answers read as words; choice answers show the option key
    shown = "\n\n".join(f"Example {i}:\n{t}\nAnswer: {words.get(a, a)}" for i, (t, a) in enumerate(examples, 1))
    return f"Labelled examples:\n\n{shown}\n\nText to classify:\n{text}"


PILOT = {"medabs": "medabs_train", "hoc": "hoc_dev", "cxr": "cxr_train"}  # never the test split


def requests(task, k_shot, seed=0, limit=None, pilot=False):
    """Yield (item id, question id, gold key, Jeff/Jev DecisionInput). pilot=True uses held-out non-test items
    (minus every item that could be drawn as a few-shot example for seeds 0-2) for checks before freezing."""
    shots = {qid: few_shot(task, qid, seed) if k_shot else [] for qid in task.questions}
    if pilot:
        used = {t for s in range(3) for qid in task.questions for t, _ in few_shot(task, qid, s)}
        items = [i for i in load(PILOT[task.name]) if i["text"] not in used][:limit]
    else:
        items = load(task.test)[:limit]
    for item in items:
        for qid, question in task.questions.items():
            yield item["id"], qid, task.gold(item, qid), {"state": state(item["text"], shots[qid]), "question": question}
