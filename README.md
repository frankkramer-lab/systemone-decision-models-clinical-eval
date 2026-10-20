# System One models on biomedical and clinical text classification

Code and results for the paper *Thinking Fast and Slow on Clinical Text: System One Models versus Reasoning LLMs*
(MIE, under review). The study compares System One models (Jev, OpenJev, Jeff) with their non-fine-tuned base models and
with LLM references (with and without reasoning) on three classification tasks:
- Medical Abstracts
- Hallmarks of Cancer (HoC) 
- MIMIC-CXR

The paper reports the zero-shot results but the one-shot results (one labeled example per class with three draws) are included here as well.

## Contents

- `runs/`: aggregated results, latency and repeatability check of Jev. In `results.csv`, `f1` is the published metric (macro-F1, for HoC micro-F1)
- `predictions/`: per-item predictions for Medical Abstracts and HoC (zero-shot and one-shot)
- `src/jt/`: This includes the evaluation code. Note that the code was LLM-generated!
- `scripts/`: model servers and the network round-trip measurement (probably not relevant to you)
- `paper/`: scripts for creating Table 1 and Figure 1. (The figure needs matplotlib!)

**MIMIC-CXR:** only aggregate scores are included.

## Reproduce the Data

**Important note:** Due to implementation details, the model responses of the System One decision models are not 100% deterministic. This also includes the commercial Jev model from TypeSafe AI. The probability scores might vary for repeated requests for the same input.

1. Python >= 3.12 with [uv](https://docs.astral.sh/uv/)
   * clone [firelex/jeff](https://github.com/firelex/jeff) at commit `3720e7c` into `third_party/jeff`
   * then run `uv sync`
   * copy `.env.example` to `.env` and fill in the endpoint, API keys and the MIMIC-CXR path you need.
2. Data:
   * Medical Abstracts (Schopf et al., CC BY-SA 3.0)
   * Hallmarks of Cancer (Baker et al., BLURB split, GPL-3.0) are prepared by `python -m jt.prepare`.
   * MIMIC-CXR requires credentialed PhysioNet access (MIMIC-CXR and MIMIC-CXR-JPG 2.1.0, radiologist-labeled test set); set `CXR_SOURCE_DIR` to the prepared files (format in `src/jt/config.py`).
3. Models:
   * Jeff v1.2 (`jeff-legacy/Jeff-Qwen3.5-0.8B`, `-2B`)
   * Qwen3.5 0.8B/2B, `openjev/openjev-FP8` with its helper (`scripts/serve_openjev.sh`)
   * Qwen3.8 27B FP8 (`scripts/serve_qwen38.sh`)
   * Hosted: Jev (TypeSafe API key),
   * DeepSeek V4 Flash 0731 (e.g. via OpenRouter key)
   * GLM 5.3 Flash (any OpenAI-compatible endpoint serving it, e.g. via OpenRouter)
4. Run:
```bash
python -m jt.run --task medabs --backend jeff-0.8b --k 0 --orders 2
python -m jt.report
python -m jt.paired
```
5. Materials:
   * Table 1: `python paper/make_table.py`
   * Figure 1: `python paper/make_figure.py`

# Citation
TODO