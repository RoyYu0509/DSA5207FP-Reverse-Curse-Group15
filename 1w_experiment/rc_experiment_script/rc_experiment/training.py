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
from rc_experiment.eval import coverage_score


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


from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# ... (other imports remain unchanged)

# Helper function to manually pad prompt for generation (no changes here)
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

def casual_llm_train(model_name, model_obj, tokenizer_obj, optimizer_obj,
                     train_loader, val_loader, device,
                     max_target_length, num_epochs,
                     patience=2, min_delta=0.0):
    """
    Train `model_obj` and apply early stopping on *validation error‑rate* (e_rate).
    An improvement means e_rate decreases by more than `min_delta`.
    """
    train_losses, val_losses, val_accuracies = [], [], []

    best_e_rate = float('inf')       # ── track *lowest* error‑rate so far
    patience_counter = 0
    saving_dir = f"./best_model/{model_name}"   # (set once; overwritten on each improvement)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # ── Training phase ────────────────────────────────────────────────
        model_obj.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model_obj(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            loss = outputs.loss

            optimizer_obj.zero_grad()
            loss.backward()
            optimizer_obj.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validation phase ──────────────────────────────────────────────
        model_obj.eval()
        val_loss_total, correct, total = 0.0, 0, 0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model_obj(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                val_loss_total += outputs.loss.item()

                # ≡≡≡ prediction / scoring (unchanged) ≡≡≡
                prompt_batch = _build_prompt_batch(input_ids, labels, tokenizer_obj)
                prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}
                preds = model_obj.generate(
                    **prompt_batch,
                    max_new_tokens=max_target_length,
                    pad_token_id=tokenizer_obj.pad_token_id,
                ).cpu()

                attn_mask_cpu = prompt_batch["attention_mask"].cpu()
                for i, gen_ids in enumerate(preds):
                    comp_ids   = _extract_completion(gen_ids, attn_mask_cpu[i])
                    pred_text  = tokenizer_obj.decode(comp_ids, skip_special_tokens=True).strip()
                    true_ids   = labels[i][labels[i] != -100]
                    true_text  = tokenizer_obj.decode(true_ids.tolist(), skip_special_tokens=True).strip()

                    total   += 1
                    correct += coverage_score(pred_text, true_text)

        avg_val_loss = val_loss_total / len(val_loader)
        val_coverage = correct / total if total > 0 else 0.0
        e_rate       = 1 - val_coverage          # ↓ lower is better
        val_losses.append(avg_val_loss)
        val_accuracies.append(e_rate)

        print(f"Epoch {epoch+1:02}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Error Rate: {e_rate*100:.2f}%")

        # ── Early‑stopping on e_rate ──────────────────────────────────────
        if e_rate < best_e_rate - min_delta:     # improvement = ↓ e_rate
            best_e_rate = e_rate
            patience_counter = 0
            model_obj.save_pretrained(saving_dir)
            tokenizer_obj.save_pretrained(saving_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered (no improvement in error‑rate).")
                break

    return saving_dir, train_losses, val_losses, val_accuracies



import os
import matplotlib.pyplot as plt
def plot_losses(train_losses, val_losses, model_name):
    """
    Plot training and validation losses over epochs, and save to a file.

    Parameters:
        train_losses (list of float): Training loss values per epoch.
        val_losses (list of float): Validation loss values per epoch.
        model_name (str): Name of the model (slashes will be replaced).
        title (str): Title for the plot.
    """
    # Sanitize model name for filename use
    safe_model_name = model_name.replace("/", "__")

    # Ensure output directory exists
    output_dir = "training_plot"
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    # Accuracy Loss = 1 - Accuracy
    plt.plot(epochs, val_losses, label='Validation Accuracy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid()

    # Save plot with sanitized model-specific filename
    plot_path = os.path.join(output_dir, f"{safe_model_name}_loss_plot.png")
    plt.savefig(plot_path)
    plt.close()