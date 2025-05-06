import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def quanti_lora_md(lora_config_dict, model_name, gradient_checkpointing=False):
    """
    Return: A quantized LoRA model. 
    
    Parameters:
        - lora_config_dict: A dictionary for basic lora configuration defined in `peft` package
            Example:
            lora_config_kwargs = {
                "r": 16,                 # LoRA rank
                "lora_alpha": 16,        # LoRA scaling factor
                "lora_dropout": 0.05,    # LoRA dropout
                "bias": "none",          # Bias handling
                "task_type": "CAUSAL_LM" # Task type
            }
        - model_name: A hugging face checkpoint model
        - gradient_checkpointing: A toogle to save memory

    Note: The LoRA method here is applied to every moudels of the mode (Every Linear Layer).
    """
    # Attempt to load the model in 8-bit mode
    # TODO: Can't load in with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            # load_in_8bit=True,
            # device_map="cuda"
            # torch_dtype=torch.float16  # use fp16 if MPS supports it
            # If you want to use 4-bit (QLoRA) instead, you could do:
            # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
            # and pass quantization_config=quantization_config (while setting load_in_8bit=False).
        )
    except Exception as e:
        raise SystemExit("❌ 8-bit quantized loading failed. Make sure you have a CUDA-compatible GPU and `bitsandbytes` installed. Aborting.") 

    # Prepare model for k-bit (here 8-bit) training – e.g., cast layer norms to float32 for stability
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if configured (saves memory at cost of compute speed)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Wrap the model's components with LoRA Adapter on all linear layers
    # These are the parameters that you want to apply LoRA on
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name.split('.')[-1])
    target_modules = list(set(target_modules))

    # Define LoRA configuration
    lora_config = LoraConfig(
        target_modules = target_modules,
        **lora_config_dict
    )

    # Apply LoRA to the base model
    lora_qt_model = get_peft_model(model, lora_config)

    # Display parameters shrinkage after LoRA
    lora_qt_model.print_trainable_parameters()
    
    return lora_qt_model