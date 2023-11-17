import os
import torch
import logging  
import argparse
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel

def merge_and_unload(base_model: str, adapter_model: str, upload_name: str, base_dtype: torch.dtype=None, device: str="cpu"):
    logging.info(f"Load base model: {base_model}")
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=base_dtype if base_dtype is not None else None
    )
    logging.info(f"Load adapter model: {adapter_model}")
    model = PeftModel.from_pretrained(base_model_reload, adapter_model, device_map=device)

    logging.info(f"Load tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    logging.info(f"Merge adapter with base model")
    model = model.merge_and_unload()

    if upload_name is None:
        upload_name = adapter_model

    logging.info(f"Upload model")
    model.push_to_hub(upload_name, use_temp_dir=False)
    tokenizer.push_to_hub(upload_name, use_temp_dir=False)

    logging.info(f"All done, model pushed to: {upload_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge and upload model to huggingface hub")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_model", type=str, required=True)
    parser.add_argument("--upload_name", type=str)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # huggingface_hub.login()

    merge_and_unload(**vars(args))
