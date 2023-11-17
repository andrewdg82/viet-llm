import os
import torch
import time
import logging  
import argparse
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel, AutoPeftModelForCausalLM

class VietLLMGenerator:
    def __init__(self, model: str, quantized: bool=True, base_dtype: torch.dtype=None, device: str="cpu"):
        logging.info(f"Loading model: {model}")
        if quantized:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=base_dtype if base_dtype not None else None
                device_map=device
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model,
                return_dict=True,
                low_cpu_mem_usage=True,
                device_map=device,
                torch_dtype=base_dtype if base_dtype not None
                trust_remote_code=True
            )

        logging.info(f"Load tokenizer: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        self.model = model.eval()
        self.tokenizer = tokenizer

    def generate(self, input: Union[str, List[str]], temperature: float=0, top_k: int=1, max_new_tokens: int=None):
        st_time = time.time()

        self.generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs = model.generate(**input, generation_config=self.generation_config)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info(f"Input: {input}")
        logging.info(f"Output: {output}")
        logging.info(f"Time cost: {time.time()-st_time}s")

        return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge and upload model to huggingface hub")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantized", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    generator = VietLLMGenerator(**vars(args))

    sample_input = "Hà Nội ở đâu?"

    generator.generate(sample_input)
