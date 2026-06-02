"""
callbacks.py
────────────
Custom HuggingFace Trainer callbacks:

  SampleGenerationCallback  — generates a qualitative example at each eval,
                              so you can watch the model learn to reason.
  RichProgressCallback      — replaces the default tqdm bar with a richer
                              Rich terminal display.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ─── Sample test question (shown during training to monitor progress) ──────────

_SAMPLE_QUESTION = (
    "A 58-year-old male with a history of hypertension and diabetes presents "
    "to the emergency department with sudden onset crushing chest pain radiating "
    "to the jaw, associated with diaphoresis and nausea. ECG shows ST elevation "
    "in leads V1-V4. Troponin is elevated at 2.4 ng/mL. "
    "What is the diagnosis and immediate management?"
)

_EXPECTED_ANSWER_KEYWORDS = ["stemi", "anterior", "pci", "lad", "catheterization"]


# ─── Callbacks ─────────────────────────────────────────────────────────────────

class SampleGenerationCallback(TrainerCallback):
    """
    At each evaluation step, run inference on a fixed clinical question
    and print the model's current reasoning chain.

    This lets you watch qualitatively how the model's reasoning evolves
    as training progresses — far more informative than loss alone.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    system_message : str
        The physician system prompt.
    question : str
        The fixed question to use for qualitative monitoring.
    max_new_tokens : int
        Limit on generated tokens per sample.
    log_to_file : bool
        If True, also append samples to a text log file.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        system_message: str,
        question: str = _SAMPLE_QUESTION,
        max_new_tokens: int = 512,
        log_to_file: bool = True,
        log_path: str = "results/sample_generations.txt",
    ) -> None:
        self.tokenizer = tokenizer
        self.system_message = system_message
        self.question = question
        self.max_new_tokens = max_new_tokens
        self.log_to_file = log_to_file
        self.log_path = log_path
        self._step_count = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[PreTrainedModel] = None,
        **kwargs,
    ) -> None:
        if model is None:
            return

        self._step_count += 1
        step = state.global_step
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE GENERATION — Step %d", step)
        logger.info("=" * 70)

        try:
            output = self._generate(model)
            self._print_sample(step, output)
            if self.log_to_file:
                self._log_to_file(step, output)
        except Exception as exc:
            logger.warning("Sample generation failed: %s", exc)

    def _generate(self, model: PreTrainedModel) -> str:
        """Run inference on the sample question."""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        model.train()

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _print_sample(self, step: int, output: str) -> None:
        border = "─" * 70
        print(f"\n{border}")
        print(f"  Step {step} | Sample generation")
        print(border)
        print(f"Q: {self.question[:120]}...")
        print(border)
        print(output[:1200])
        if len(output) > 1200:
            print(f"[... {len(output) - 1200} more characters]")
        print(f"{border}\n")

        # Check if expected keywords appear
        output_lower = output.lower()
        hits = [kw for kw in _EXPECTED_ANSWER_KEYWORDS if kw in output_lower]
        if hits:
            print(f"  ✓ Keywords found: {hits}")
        else:
            print(f"  ✗ None of expected keywords found: {_EXPECTED_ANSWER_KEYWORDS}")
        print()

    def _log_to_file(self, step: int, output: str) -> None:
        import os
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"STEP {step} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Q: {self.question}\n\n")
            f.write(f"A:\n{output}\n")


class RichProgressCallback(TrainerCallback):
    """
    Replaces default HuggingFace progress output with a cleaner Rich display.
    Logs loss, learning rate, and epoch at each logging step.
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._start_time = time.time()
        logger.info("Training started at %s", time.strftime("%H:%M:%S"))

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return

        elapsed = (time.time() - (self._start_time or time.time())) / 60
        step = state.global_step
        max_steps = state.max_steps

        loss = logs.get("loss", logs.get("train_loss", "—"))
        eval_loss = logs.get("eval_loss", "—")
        lr = logs.get("learning_rate", "—")
        epoch = logs.get("epoch", state.epoch or "—")

        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        eval_str = f"{eval_loss:.4f}" if isinstance(eval_loss, float) else str(eval_loss)
        lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
        epoch_str = f"{epoch:.2f}" if isinstance(epoch, float) else str(epoch)

        pct = int(100 * step / max_steps) if max_steps else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)

        print(
            f"\r  [{bar}] {pct:3d}%  "
            f"step={step}/{max_steps}  "
            f"epoch={epoch_str}  "
            f"loss={loss_str}  "
            f"eval_loss={eval_str}  "
            f"lr={lr_str}  "
            f"t={elapsed:.1f}m",
            flush=True,
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        elapsed = (time.time() - (self._start_time or time.time())) / 60
        logger.info("Training finished | total time: %.1f minutes", elapsed)