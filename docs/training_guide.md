# Training Guide

## Overview

This guide covers everything needed to reproduce the training run from scratch.

---

## Hardware Requirements

| Hardware | VRAM | Est. Time (5K samples, 3 epochs) | Status |
|---|---|---|---|
| Google Colab T4 | 16 GB | ~4 hours | ✅ Recommended |
| Google Colab A100 | 40 GB | ~1.5 hours | ✅ Fastest |
| Local RTX 3090/4090 | 24 GB | ~3.5 hours | ✅ Works |
| Local RTX 3080 | 10 GB | ❌ OOM | Not supported |
| CPU only | — | Days | ❌ Not feasible |

---

## Step-by-Step: Google Colab

1. Open `notebooks/02_training_colab.ipynb`
2. Runtime → Change runtime type → **T4 GPU**
3. Run Cell 1 (installs dependencies) — wait ~3 minutes
4. Run Cell 2 (mounts Google Drive) — optional but recommended for saving checkpoints
5. Run Cell 3 onwards — follows the same pipeline as `scripts/train.py`

---

## Step-by-Step: Local Training

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/medical-reasoning-llm.git
cd medical-reasoning-llm
pip install -e ".[dev]"

# 2. Set up environment
cp .env.example .env
# Add your HF_TOKEN to .env

# 3. Sanity check (no training, just loads everything)
python scripts/train.py --dry_run

# 4. Train with defaults (5K samples, 3 epochs)
python scripts/train.py

# 5. Train with overrides
python scripts/train.py \
  --num_samples 2000 \
  --num_epochs 1 \
  --output_dir ./results/quick_run
```

---

## Hyperparameter Reference

### Critical Parameters

| Parameter | Default | Notes |
|---|---|---|
| `lora.r` | 16 | LoRA rank. Higher = more capacity but more VRAM. Try 8 for speed. |
| `lora.lora_alpha` | 32 | Scaling: always 2× rank. |
| `training.learning_rate` | 2e-4 | Standard for QLoRA. Don't go above 3e-4. |
| `training.num_train_epochs` | 3 | 3 is optimal; >4 often overfits. |
| `training.per_device_train_batch_size` | 2 | T4 max with seq_len=2048. |
| `training.gradient_accumulation_steps` | 4 | Effective batch = 8. |
| `data.max_seq_length` | 2048 | CoT reasoning chains can be long. |

### Reducing VRAM Usage (if OOM)

1. Reduce `max_seq_length` to 1024 (will filter out some long samples)
2. Set `lora.r = 8` (fewer trainable params)
3. Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` to 8

### Improving Quality

1. Use all 25K samples (set `num_samples: 25000` — requires ~16hrs on T4)
2. Increase `lora.r` to 32 if using A100
3. Try `num_train_epochs: 5` with early stopping

---

## Expected Training Output

```
╔══════════════════════════════════════════════════════════╗
║        Medical Reasoning LLM — QLoRA Training            ║
╚══════════════════════════════════════════════════════════╝

00:00:01 | INFO     | Loading config: config/training_config.yaml
00:00:15 | INFO     | Full dataset size: 25000 samples
00:00:16 | INFO     | Using 5000 / 25000 samples (seed=42)
00:01:30 | INFO     | Loading base model: Qwen/Qwen2.5-3B-Instruct
00:02:45 | INFO     | QLoRA: 4-bit NF4 | double_quant=True | compute_dtype=float16
00:03:00 | INFO     | Attaching LoRA: r=16 | alpha=32 | scaling=2.00
00:03:01 | INFO     | Trainable params: 23,592,960 (0.7614%)
00:03:30 | INFO     | Step 5/5 — Training

  [████░░░░░░░░░░░░░░░░]  20%  step=100/500  epoch=0.25  loss=1.2341  ...
  [████████░░░░░░░░░░░░]  40%  step=200/500  epoch=0.50  loss=0.9823  ...
```

---

## Monitoring Training Quality

Every 100 steps, `SampleGenerationCallback` generates a clinical scenario and
prints the model's current reasoning. Watch for:

- **Early training (steps 0–100):** Outputs often rambling or off-topic
- **Mid training (steps 200–350):** Structure emerges; medical terms appear
- **Late training (steps 400–500):** Clear reasoning chains; correct diagnoses

---

## Checkpoints and Recovery

Checkpoints are saved every 200 steps to `results/checkpoints/`.
To resume from a checkpoint:

```bash
# The trainer automatically loads the best checkpoint at the end.
# To resume mid-run, point output_dir to the existing results dir:
python scripts/train.py --output_dir results/run_01
```

---

## Pushing to HuggingFace Hub

```yaml
# In config/training_config.yaml:
training:
  push_to_hub: true
  hub_model_id: "YOUR_USERNAME/medical-reasoning-qwen2.5-3b-qlora"
```

Then make sure `HF_TOKEN` is set in `.env`.