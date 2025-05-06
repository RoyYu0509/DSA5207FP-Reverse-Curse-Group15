from __future__ import annotations
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import random
import math
from tqdm import tqdm
import pandas as pd


def _extract_completion(gen_ids, prompt_mask_row):
    """
    Slice the generated sequence right after the last prompt token.

    Parameters
    ----------
    gen_ids : 1‑D LongTensor
        Full sequence returned by `model.generate()` (prompt + completion).
    prompt_mask_row : 1‑D LongTensor
        The attention‑mask row that was fed to the model for that prompt.
        1 = real token (prompt), 0 = pad.

    Returns
    -------
    str
        Decoded completion text (no prompt, no pads, no special tokens).
    """
    prompt_len = int(prompt_mask_row.sum().item())   # number of 1s
    completion_ids = gen_ids[prompt_len:]            # everything *after* the prompt
    return completion_ids


def _build_prompt_batch(input_ids, labels, tokenizer):
    """
    • Keeps existing left pads
    • Removes everything to the right of the prompt
    • Re-pads (on the left) so the batch is rectangular again
    Returns a dict ready for model.generate().
    """
    prompt_only = []
    for seq, lab in zip(input_ids, labels):
        first_comp = (lab != -100).nonzero(as_tuple=True)[0][0].item()
        prompt_only.append(seq[:first_comp])  # ← no right pads! (prompt portion, includes instruction if present)
    return tokenizer.pad(
        {"input_ids": prompt_only},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import pandas as pd
from tqdm import tqdm

def rc_eval(test_loader,
            model_obj,
            tokenizer_obj,
            device,
            max_input_length: int,
            max_target_length: int,
            instruction: str = None):
    """
    Evaluate `model_obj` on `test_loader` and return a DataFrame containing:
        Prompt, Prediction, Ground-Truth, Exact Match ✅/❌

    * Prompt      —— only the user prompt (instruction stripped; true completion never included)
    * Prediction  —— model‐generated completion only
    * Ground-Truth—— reference completion from the dataset
    """
    # 1) prepare tokenizer & model
    tokenizer_obj.padding_side = "left"
    model_obj.eval()

    # 2) measure instruction length if provided
    instr_len = 0
    if instruction:
        instr_ids = tokenizer_obj.encode(instruction, add_special_tokens=False)
        instr_len = len(instr_ids)

    correct, total = 0, 0
    rows = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # build prompt-only (instr + user prompt)
            prompt_batch = _build_prompt_batch(input_ids, labels, tokenizer_obj)
            prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}

            # generate [instr + prompt] + new tokens
            preds = model_obj.generate(
                **prompt_batch,
                max_new_tokens=max_target_length,
                pad_token_id=tokenizer_obj.pad_token_id,
                # temperature=0.1, 
            )

            attn_mask_cpu = prompt_batch["attention_mask"].cpu()   # bring once to CPU

            for i, gen_ids in enumerate(tqdm(preds, desc="Decoding predictions", leave=False, unit="sample")):
                # ----- completion text -----
                comp_ids = _extract_completion(gen_ids, attn_mask_cpu[i])
                pred_text = tokenizer_obj.decode(comp_ids, skip_special_tokens=True).strip()

                # ----- user prompt text (after instruction) -----
                nonpad_prompt_ids = input_ids[i][attention_mask[i] == 1]
                prompt_text = tokenizer_obj.decode(nonpad_prompt_ids[instr_len:], skip_special_tokens=True).strip()

                # ----- ground truth -----
                true_ids   = labels[i][labels[i] != -100]
                true_text  = tokenizer_obj.decode(true_ids.tolist(), skip_special_tokens=True).strip()

                # coverage
                total   += 1
                correct += coverage_score(pred_text, true_text)

                rows.append({"Prompt": prompt_text,
                            "Prediction": pred_text,
                            "Ground-Truth": true_text})

    coverage = correct / total if total else 0.0
    print(f"Test Coverage Accuracy: {coverage*100:.2f}% ({correct}/{total})")

    return pd.DataFrame(rows)



from typing import List
import spacy
from rapidfuzz import fuzz

"""
Metrics Definition: **https://chatgpt.com/c/68173af1-83b4-800b-825c-62a84b008fdb**


Smoothed “key‑information coverage” metric
=========================================

Returns a value in the closed interval [0.0, 1.0].

Scoring logic
-------------
1.  **Extract key phrases** (noun‑phrases + named entities) from the *expected*
    text with spaCy.
2.  **If at least one phrase is found**
        score = average_i  fuzzy_partial_ratio(phrase_i, prediction) / 100
    (i.e. recall of the set of phrases, softened by fuzzy matching)
3.  **If no phrases are extracted**  
        score = fuzzy_partial_ratio(expected, prediction) / 100
4.  An *exact* textual match of the two strings always returns 1.0.

Dependencies
------------
```bash
pip install spacy rapidfuzz
python -m spacy download en_core_web_sm
```
"""

# ── spaCy pipeline (keep the parser so noun_chunks work) ──
_NLP = spacy.load("en_core_web_sm")

_STOP_WORDS = _NLP.Defaults.stop_words
_MIN_PHRASE_LEN = 2        # ignore 1‑word (likely non‑salient) phrases
_FUZZ_THRESHOLD = 85       # threshold used *within* per‑phrase matching



# ---------------------------------------------------------------------
# Helper: extract salient phrases from a string
# ---------------------------------------------------------------------
def _extract_phrases(text: str) -> List[str]:
    """
    Return unique, lower‑cased noun‑phrases + named entities,
    stripped of punctuation / stop‑words at their ends.
    """
    doc = _NLP(text)

    raw_spans = [span.text for span in doc.ents] + [chunk.text for chunk in doc.noun_chunks]
    phrases: set[str] = set()

    for span_text in raw_spans:
        # Re‑tokenise the span to trim stop‑words / punctuation
        tokens = [
            tok.text.lower()
            for tok in _NLP(span_text)
            if not (tok.is_punct or tok.text.lower() in _STOP_WORDS)
        ]
        if len(tokens) >= _MIN_PHRASE_LEN:
            phrases.add(" ".join(tokens))

    return list(phrases)

# ---------------------------------------------------------------------
# Public metric
# ---------------------------------------------------------------------
def coverage_score(
    prediction: str,
    expected: str,
    *,
    phrase_threshold: int = _FUZZ_THRESHOLD,
) -> float:
    """
    Parameters
    ----------
    prediction : str
        Text generated by the model.
    expected : str
        Reference text that contains the key information.
    phrase_threshold : int, optional
        Minimum RapidFuzz partial‑ratio (0‑100) that counts a phrase as matched.

    Returns
    -------
    float
        A score in [0.0, 1.0] indicating how well *prediction* covers the
        information in *expected*.
    """
    pred_lc = prediction.lower()
    exp_lc = expected.lower().strip()

    # exact match short‑circuit
    if pred_lc.strip() == exp_lc:
        return 1.0

    # 1 — phrase‑based recall if we find any noun‑phrases / entities
    phrases = _extract_phrases(expected)
    if phrases:
        fuzzy_scores = [
            fuzz.partial_ratio(phrase, pred_lc) / 100.0
            for phrase in phrases
        ]
        # treat scores below the per‑phrase threshold as 0
        recall_hits = [
            s if s * 100 >= phrase_threshold else 0.0
            for s in fuzzy_scores
        ]
        return sum(recall_hits) / len(phrases)

    # 2 — fallback to whole‑string similarity when no phrases extracted
    return fuzz.partial_ratio(exp_lc, pred_lc) / 100.0
