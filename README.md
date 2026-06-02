<div align="center">

# 🩺 Medical-Reasoning-LLM

### Teaching Language Models *How to Think* in the Clinic

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-3B](https://img.shields.io/badge/Base%20Model-Qwen2.5--3B-purple)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-medical--o1--reasoning--SFT-orange)](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/02_training_colab.ipynb)

</div>

---

> **Most medical NLP extracts labels. This project teaches a model to reason.**
>
> We fine-tune `Qwen2.5-3B-Instruct` using **QLoRA** on 5,000 physician-written
> chain-of-thought diagnostic scenarios. The result: a compact model that doesn't
> just predict — it *explains its clinical thinking*, step by step.

---

## 📋 Table of Contents

- [Motivation](#-motivation)
- [What Makes This Different](#-what-makes-this-different)
- [Architecture](#-architecture)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 💡 Motivation

During development of [Second Eye Nepal](https://www.secondeye.com.np/)'s **ChestGuru** clinical
decision support system, a recurring problem emerged: small language models can extract
named entities (drug names, diagnoses, symptoms) but they cannot *reason* about them.
A physician reading an ECG doesn't just label it — they construct a causal narrative:

> *"ST elevation in leads II, III, aVF → inferior wall involvement → RCA occlusion
> territory → 80% probability → immediate PCI window..."*

This project directly addresses that gap: **instruction-tuning a 3-billion-parameter model
to produce explicit diagnostic reasoning chains**, not just final-answer labels.

The implications for low-resource clinical settings (rural Nepal, sub-Saharan Africa) are
significant — a model that explains its reasoning is a model that a clinician can
*audit*, *trust*, and *override*.

---

## 🔬 What Makes This Different

| Approach | Traditional Medical NLP | **This Project** |
|---|---|---|
| Task type | Named Entity Recognition | Chain-of-Thought Reasoning |
| Output | Structured labels | Full reasoning → answer |
| Model need | Any ML model | LLM capability required |
| Auditability | Black box | Step-by-step explanation |
| Clinical trust | Low | Verifiable |
| Trainable on T4 | ✅ | ✅ (via QLoRA) |
| Reproducible | ✅ | ✅ (public dataset + configs) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEDICAL REASONING PIPELINE                        │
│                                                                      │
│   Clinical Scenario                                                  │
│   ───────────────                                                    │
│   "45yo male, crushing chest pain, ST elevation II/III/aVF..."      │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────────┐     ┌──────────────────────────────────┐      │
│   │   System Prompt  │────▶│   Qwen2.5-3B-Instruct + QLoRA  │      │
│   │   (Physician     │     │                                  │      │
│   │    Persona)      │     │   Base: 4-bit NF4 quantization  │      │
│   └─────────────────┘     │   Adapter: LoRA r=16, α=32      │      │
│                            │   Trainable params: ~24M / 3B   │      │
│                            └──────────────────────────────────┘      │
│                                       │                              │
│                                       ▼                              │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │                  REASONING CHAIN OUTPUT                   │      │
│   │                                                           │      │
│   │  Step 1: Anatomical localization (inferior wall STEMI)   │      │
│   │  Step 2: Vascular territory (RCA, 80% of cases)          │      │
│   │  Step 3: Time-sensitive decision (PCI window < 90 min)   │      │
│   │  Step 4: Differentials ruled out (NSTEMI, Pericarditis)  │      │
│   │  ─────────────────────────────────────────────────────   │      │
│   │  Final Answer: Inferior STEMI — RCA occlusion            │      │
│   │               Immediate PCI indicated                    │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Training Setup:
  Dataset  : medical-o1-reasoning-SFT (5K/25K samples, stratified)
  Method   : QLoRA — bitsandbytes 4-bit NF4 + LoRA adapters
  Trainer  : HuggingFace TRL SFTTrainer
  Hardware : NVIDIA T4 (Google Colab) — 6 hrs end-to-end
  Epochs   : 3 | Batch: 2 per device, grad accum 4 (effective: 8)
```

---

## 📊 Results

> Results are populated after training. Run `scripts/evaluate.py` to reproduce.

| Metric | Base Qwen2.5-3B | Fine-tuned (ours) | Δ |
|---|---|---|---|
| Answer Accuracy (exact match) | — | — | — |
| ROUGE-L vs. Gold CoT | — | — | — |
| Clinical Coherence (GPT-4 judge, 1–5) | — | — | — |
| Avg. Reasoning Chain Length (tokens) | — | — | — |

*Full results, loss curves, and qualitative examples in [`results/`](results/).*

---

## ⚡ Quick Start

**Inference only (no training required):**

```python
from src.medical_reasoning.inference.pipeline import MedicalReasoningPipeline

pipeline = MedicalReasoningPipeline.from_pretrained(
    base_model="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="./results/final_adapter",   # after training
    load_in_4bit=True,
)

result = pipeline.reason(
    "A 67-year-old woman presents with sudden onset severe headache "
    "she describes as 'the worst headache of my life'. She is afebrile. "
    "CT head is negative for hemorrhage. What is the diagnosis?"
)

print(result.reasoning_chain)
print(result.final_answer)
```

---

## 🛠️ Installation

### Option A — Local (GPU required, 8 GB+ VRAM for inference)

```bash
git clone https://github.com/YOUR_USERNAME/medical-reasoning-llm.git
cd medical-reasoning-llm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Option B — Google Colab (Recommended for training)

Open [`notebooks/02_training_colab.ipynb`](notebooks/02_training_colab.ipynb) directly.
Runtime → Change runtime type → **T4 GPU**. Everything installs in Cell 1.

### Environment Variables

```bash
cp .env.example .env
# Fill in:
#   HF_TOKEN=<your_huggingface_token>   # needed to push adapter to Hub
#   WANDB_API_KEY=<optional>            # for experiment tracking
```

---

## 📦 Dataset

We use a 5,000-sample stratified subset of
[`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
— a dataset of 25,000 physician-quality medical QA pairs with **full chain-of-thought
reasoning**, created by the HuatuoGPT-o1 team.

**Sample structure:**

```json
{
  "question": "A 45-year-old male presents with crushing chest pain radiating
               to the left arm for 90 minutes. BP 90/60, HR 110. ECG shows
               ST elevation in leads II, III, aVF with reciprocal depression
               in I and aVL. What is the diagnosis and immediate management?",

  "complex_cot": "Let me analyze this systematically. The patient presents with
                  classic ischemic chest pain symptoms (crushing quality, radiation
                  to left arm). The hemodynamic instability (hypotension, tachycardia)
                  suggests significant myocardial compromise. The ECG findings are
                  diagnostic: ST elevation in II, III, aVF identifies inferior wall
                  STEMI. The inferior wall is supplied by the right coronary artery
                  (RCA) in 80% of patients. Reciprocal changes in I and aVL confirm
                  this is true elevation, not artifact. The 90-minute symptom duration
                  means we are within the PCI window...",

  "response": "Inferior STEMI secondary to RCA occlusion. Immediate management:
               aspirin 325mg + P2Y12 inhibitor, activate cath lab, primary PCI
               within 90 minutes of first medical contact. Consider right-sided
               leads to rule out RV infarct."
}
```

**Data split (auto-generated, reproducible via seed=42):**

| Split | Samples |
|---|---|
| Train | 4,000 |
| Validation | 500 |
| Test (held out) | 500 |

To explore the dataset interactively, open [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).

---

## 🚂 Training

### Via script (local GPU):

```bash
python scripts/train.py \
  --config config/training_config.yaml \
  --output_dir results/run_01
```

### Via Colab notebook:

Open [`notebooks/02_training_colab.ipynb`](notebooks/02_training_colab.ipynb) — 14 cells,
end-to-end in ~4 hours on T4.

### Key hyperparameters (`config/training_config.yaml`):

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4        # effective batch size: 8
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  max_seq_length: 2048
  fp16: true
```

Full config reference: [`docs/training_guide.md`](docs/training_guide.md)

---

## 📐 Evaluation

```bash
python scripts/evaluate.py \
  --adapter_path results/run_01/final_adapter \
  --test_data data/test.json \
  --output results/eval_report.json
```

**Metrics computed:**

- `answer_accuracy` — exact match on final answer (after reasoning chain)
- `rouge_l` — ROUGE-L of generated CoT against gold CoT
- `chain_completeness` — fraction of outputs with >3 reasoning steps
- `avg_chain_length` — mean token count of reasoning section

See [`docs/evaluation_guide.md`](docs/evaluation_guide.md) for the GPT-4 judge rubric used for clinical coherence scoring.

---

## 🔮 Inference

```bash
# Single query
python scripts/infer.py \
  --adapter_path results/run_01/final_adapter \
  --question "Your clinical question here"

# Batch inference from file
python scripts/infer.py \
  --adapter_path results/run_01/final_adapter \
  --input_file data/custom_questions.json \
  --output_file results/predictions.json
```

---

## 🗂️ Project Structure

```
medical-reasoning-llm/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── config/
│   ├── training_config.yaml      # All training hyperparameters
│   ├── model_config.yaml         # Model & quantization settings
│   └── eval_config.yaml          # Evaluation settings
│
├── src/
│   └── medical_reasoning/
│       ├── data/
│       │   ├── dataset.py         # HuggingFace dataset loading & caching
│       │   ├── preprocessor.py    # Chat template formatting, tokenization
│       │   └── utils.py           # Stratified splitting, statistics
│       │
│       ├── models/
│       │   ├── base.py            # Model loading with BitsAndBytes config
│       │   └── qlora.py           # LoRA attachment, adapter save/load
│       │
│       ├── training/
│       │   ├── trainer.py         # SFTTrainer wrapper with callbacks
│       │   └── callbacks.py       # Logging, early stopping, sample generation
│       │
│       ├── evaluation/
│       │   ├── metrics.py         # ROUGE-L, accuracy, chain quality
│       │   └── evaluator.py       # Full eval loop with report generation
│       │
│       └── inference/
│           └── pipeline.py        # Clean inference API
│
├── scripts/
│   ├── train.py                   # CLI training entry point
│   ├── evaluate.py                # CLI evaluation entry point
│   └── infer.py                   # CLI inference entry point
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Dataset statistics & sample viewer
│   ├── 02_training_colab.ipynb    # Full training notebook (T4, 6 hrs)
│   └── 03_evaluation.ipynb        # Results analysis & visualizations
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluation.py
│
├── docs/
│   ├── training_guide.md
│   └── evaluation_guide.md
│
└── results/
    └── .gitkeep                   # Populated after training
```

---

## 🗺️ Roadmap

- [x] Phase 1 — Project scaffold, configs, README
- [x] Phase 2 — Data pipeline (loading, preprocessing, chat templating)
- [x] Phase 3 — Model + QLoRA training pipeline
- [x] Phase 4 — Evaluation (ROUGE-L, accuracy, GPT-4 judge)
- [x] Phase 5 — Colab notebook (14-cell end-to-end)
- [ ] Phase 6 — Push adapter to HuggingFace Hub
- [ ] Phase 7 — Gradio demo
- [ ] Extend to Qwen2.5-7B with A100

---

## 📎 Citation

If you use this work, please cite:

```bibtex
@misc{medicalreasoningllm2025,
  title        = {Medical-Reasoning-LLM: QLoRA Instruction Tuning for Clinical Chain-of-Thought},
  author       = {Sandip Acharya},
  year         = {2025},
  howpublished = {\url{https://github.com/SandipAcharya/medical-reasoning-llm}},
  note         = {Fine-tuned Qwen2.5-3B-Instruct on medical-o1-reasoning-SFT dataset}
}
```

---

## 🙏 Acknowledgements

- **[FreedomIntelligence / HuatuoGPT team](https://github.com/FreedomIntelligence)** —
  for the `medical-o1-reasoning-SFT` dataset and the inspiration from HuatuoGPT-o1.
- **[Qwen Team (Alibaba DAMO)](https://huggingface.co/Qwen)** — for the Qwen2.5-3B-Instruct base model.
- **[Tim Dettmers et al.](https://arxiv.org/abs/2305.14314)** — for QLoRA.
- **[HuggingFace TRL Team](https://github.com/huggingface/trl)** — for SFTTrainer.
- **Second Eye Nepal / ChestGuru** — the clinical decision support context that motivated this work.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Medical Disclaimer:** This model is a research prototype. It is **not** validated for
> clinical use and **must not** be used for real patient care decisions. All outputs
> should be reviewed by licensed healthcare professionals.

---
