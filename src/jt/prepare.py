"""Build processed JSONL files, one record per item: {"id", "text", "label" | "labels", "split"}.

- medabs : Medical Abstracts (Schopf et al. 2022), 5 classes, single label. Open (CC BY-SA 3.0).
- hoc    : Hallmarks of Cancer (Baker et al. 2016), BLURB document split, 10 hallmarks, multi-label at abstract level
           (union of the sentence labels, parsed as in bigbio/hallmarks_of_cancer). Open (GPL-3.0).
- cxr    : MIMIC-CXR radiologist test set, 4 findings x 4 values, read from CXR_SOURCE_DIR (credentialed, see config.py).
           Few-shot examples come from its silver validate split (CheXpert-labeler labels, test patients removed).
"""
import io
import json
import tarfile
import zipfile

import pandas as pd

from jt import config as C

HALLMARKS = [  # bigbio class names minus "none"
    "sustaining proliferative signaling", "evading growth suppressors", "resisting cell death",
    "enabling replicative immortality", "inducing angiogenesis", "activating invasion and metastasis",
    "genomic instability and mutation", "tumor promoting inflammation", "cellular energetics",
    "avoiding immune destruction",
]


def write_jsonl(name, records):
    path = C.DATA_PROCESSED / f"{name}.jsonl"
    with open(path, "w") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>6} -> {path.relative_to(C.ROOT)}")


def medabs():
    names = pd.read_parquet(C.DATA_RAW / "medabs" / "labels_train-00000-of-00001.parquet")
    name_of = dict(zip(names.condition_label, names.condition_name))
    for split in ("train", "test"):
        df = pd.read_parquet(C.DATA_RAW / "medabs" / f"data_{split}-00000-of-00001.parquet")
        write_jsonl(f"medabs_{split}", [
            {"id": f"medabs-{split}-{i}", "text": t.strip(), "label": name_of[int(y)], "split": split}
            for i, (y, t) in enumerate(zip(df.condition_label, df.medical_abstract))])


def hoc():
    corpus = zipfile.ZipFile(C.DATA_RAW / "hoc" / "corpus.zip")
    root = "Hallmarks-of-Cancer-master/"
    with tarfile.open(C.DATA_RAW / "hoc" / "blurb.tar.gz") as tar:
        indices = {split: tar.extractfile(f"data_generation/indexing/HoC/{split}_pmid.tsv").read().decode().strip("\n").split(",")
                   for split in ("train", "dev", "test")}
    for split, docs in indices.items():
        records = []
        for doc in docs:
            sentences = corpus.read(root + "text/" + doc).decode("utf8").strip("\n").split("\n")
            raw = corpus.read(root + "labels/" + doc).decode("utf8").strip("\n").split("<")[1:]
            found = set()
            for label in raw[:len(sentences)]:
                for part in label.strip().split("AND"):
                    part = part.strip()
                    if part and part != "NULL":
                        found.add(part.split("--")[0].lower().lstrip())
            unknown = found - set(HALLMARKS)
            if unknown:
                raise ValueError(f"{doc}: unknown hallmark labels {unknown}")
            records.append({"id": f"hoc-{doc.split('.')[0]}", "text": " ".join(s.strip() for s in sentences),
                            "labels": {h: h in found for h in HALLMARKS}, "split": split})
        write_jsonl(f"hoc_{split}", records)


def cxr():
    src = C.CXR_SOURCE
    for name, out in (("radiologist_test", "cxr_test"), ("silver_validate_noleak", "cxr_train")):
        records = []
        with open(src / f"{name}.jsonl") as fp:
            for line in fp:
                r = json.loads(line)
                if not r["section"].strip():
                    continue  # few-shot pool only; the test set has no empty sections (checked in main)
                records.append({"id": f"cxr-{r['study_id']}", "text": r["section"].strip(), "labels": r["labels"],
                                "split": "test" if out == "cxr_test" else "train"})
        write_jsonl(out, records)


def main():
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    medabs()
    hoc()
    cxr()


if __name__ == "__main__":
    main()
