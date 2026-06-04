"""
pipeline.py
───────────
Clean, production-ready inference API for the fine-tuned
medical reasoning model. Handles model loading, prompt construction,
generation, and output parsing in a single object.

Example
-------
>>> pipeline = MedicalReasoningPipeline.from_pretrained(
...     base_model="Qwen/Qwen2.5-3B-Instruct",
...     adapter_path="./results/final_adapter",
...     load_in_4bit=True,
... )
>>> result = pipeline.reason("45yo male, crushing chest pain, ST elevation...")
>>> print(result.reasoning_chain)
>>> print(result.final_answer)
>>> print(result.confidence_note)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ─── Output container ──────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    """
    Structured output from the MedicalReasoningPipeline.

    Attributes
    ----------
    question : str
        The original input question.
    full_output : str
        Raw model output (reasoning + answer).
    reasoning_chain : str
        Parsed reasoning chain only.
    final_answer : str
        Parsed final answer only.
    generation_time_s : float
        Seconds taken to generate.
    num_tokens_generated : int
        Number of tokens in the raw output.
    confidence_note : str
        Standard disclaimer appended to all clinical outputs.
    """

    question: str
    full_output: str
    reasoning_chain: str
    final_answer: str
    generation_time_s: float = 0.0
    num_tokens_generated: int = 0
    confidence_note: str = field(
        default=(
            "⚠️  RESEARCH PROTOTYPE — Not validated for clinical use. "
            "All outputs must be reviewed by a licensed healthcare professional."
        )
    )

    def __str__(self) -> str:
        return (
            f"\n{'─'*70}\n"
            f"CLINICAL QUESTION\n{self.question}\n\n"
            f"{'─'*70}\n"
            f"REASONING CHAIN\n{self.reasoning_chain}\n\n"
            f"{'─'*70}\n"
            f"FINAL ANSWER\n{self.final_answer}\n\n"
            f"{'─'*70}\n"
            f"{self.confidence_note}\n"
            f"Generated {self.num_tokens_generated} tokens in {self.generation_time_s:.1f}s\n"
        )


# ─── Pipeline ──────────────────────────────────────────────────────────────────

class MedicalReasoningPipeline:
    """
    End-to-end inference pipeline for the fine-tuned medical reasoning model.

    Loads base model + LoRA adapter (or a merged model), constructs the
    chat-template prompt, generates, and parses the output.

    Use `from_pretrained()` as the constructor.
    """

    DEFAULT_SYSTEM_MESSAGE = (
        "You are an expert physician with deep knowledge of internal medicine, "
        "cardiology, neurology, and emergency medicine. When presented with a "
        "clinical scenario, you reason through it systematically before providing "
        "your final answer. Your reasoning should include: symptom analysis, "
        "relevant anatomy and physiology, differential diagnosis with reasoning, "
        "and evidence-based management. Always separate your thinking process "
        "from your final answer."
    )

    REASONING_HEADER = "Let me reason through this step by step:"
    ANSWER_HEADER = "Final Answer:"

    def __init__(
        self,
        model,
        tokenizer,
        system_message: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[Union[str, Path]] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        system_message: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> "MedicalReasoningPipeline":
        """
        Load the model and tokenizer, optionally with a LoRA adapter.

        Parameters
        ----------
        base_model : str
            HuggingFace model identifier for the base model.
        adapter_path : str | Path | None
            Path to the saved LoRA adapter directory.
            If None, runs the base model (for comparison).
        load_in_4bit : bool
            Use 4-bit NF4 quantization (recommended for T4).
        device_map : str
            Device map for model placement.
        system_message : str | None
            Override the default physician system prompt.
        max_new_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        MedicalReasoningPipeline
        """
        logger.info("Loading tokenizer from: %s", base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model: %s", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if not load_in_4bit else None,
            device_map=device_map,
            trust_remote_code=True,
            offload_folder="offload",
        )
        model.config.use_cache = True  # Enable KV cache for inference

        if adapter_path is not None:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from: %s", adapter_path)
            # autocast_adapter_dtype=False prevents PEFT from calling .to() on the model!
            model = PeftModel.from_pretrained(
                model, 
                str(adapter_path), 
                autocast_adapter_dtype=False,
                offload_folder="offload"
            )
            logger.info("Adapter loaded")

        # CRITICAL FIX: Manually move all non-quantized layers (Embeddings, LayerNorms, LoRA Adapters) to GPU 0
        logger.info("Forcing non-quantized layers to GPU to bypass accelerate bugs...")
        for name, param in model.named_parameters():
            if str(param.device) == "cpu" or str(param.device) == "meta":
                if not hasattr(param, "quant_state") and param.dtype != torch.uint8:
                    try:
                        param.data = param.data.to("cuda:0")
                    except Exception:
                        pass
        for name, buffer in model.named_buffers():
            if str(buffer.device) == "cpu" or str(buffer.device) == "meta":
                try:
                    buffer.data = buffer.data.to("cuda:0")
                except Exception:
                    pass

        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            system_message=system_message,
            max_new_tokens=max_new_tokens,
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    def reason(self, question: str) -> ReasoningResult:
        """
        Generate a reasoning chain and final answer for a clinical question.

        Parameters
        ----------
        question : str
            The clinical scenario or question.

        Returns
        -------
        ReasoningResult
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question},
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
        ).to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        full_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        num_tokens = len(new_tokens)

        return ReasoningResult(
            question=question,
            full_output=full_output,
            reasoning_chain=self._extract_reasoning(full_output),
            final_answer=self._extract_answer(full_output),
            generation_time_s=round(elapsed, 2),
            num_tokens_generated=num_tokens,
        )

    def reason_stream(self, question: str):
        """
        Stream tokens one-by-one as the model generates them (like DeepSeek/Claude).
        Yields raw text chunks as they are produced.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 1.0,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in a background thread so we can stream from the main thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()


    def reason_batch(self, questions: List[str]) -> List[ReasoningResult]:
        """
        Run inference on multiple questions sequentially.

        Parameters
        ----------
        questions : list of str

        Returns
        -------
        list of ReasoningResult
        """
        return [self.reason(q) for q in questions]

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _extract_answer(self, text: str) -> str:
        if self.ANSWER_HEADER in text:
            return text.split(self.ANSWER_HEADER, 1)[-1].strip()
        return text.strip().split("\n\n")[-1].strip()

    def _extract_reasoning(self, text: str) -> str:
        start = self.REASONING_HEADER
        end = self.ANSWER_HEADER
        if start in text and end in text:
            return text.split(start, 1)[-1].split(end, 1)[0].strip()
        if end in text:
            return text.split(end, 1)[0].strip()
        return text.strip()

    def __repr__(self) -> str:
        return (
            f"MedicalReasoningPipeline("
            f"max_new_tokens={self.max_new_tokens}, "
            f"do_sample={self.do_sample})"
        )