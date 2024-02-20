import torch
import time
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig, pipeline


class VietLLMGenerator:
    def __init__(self, model: str, quantized: bool=True, base_dtype: torch.dtype=torch.bfloat16, device: str="cpu"):
        print(f"Loading model: {model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        max_memory = {0: "23GiB", "cpu": "40GiB"}

        _model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config if quantized else None,
            torch_dtype=base_dtype,
            device_map={'':device},
            trust_remote_code=True,
            use_auth_token=False,
            max_memory=max_memory
        )

        print(f"Load tokenizer: {model}")

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer.padding_side = "right"

        pipe = pipeline(
            "text-generation", 
            model=_model, 
            tokenizer=tokenizer, 
            torch_dtype=base_dtype, 
            device_map={'':device}
        )

        self.model = _model
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.device = device
    
    def format_input(self, input: str) -> str:
        message = [
                {"role": "system", "content": "You are helpful assistant."},
                {"role": "user", "content": input},
            ]
        output = self.tokenizer.apply_chat_template(message, return_tensors="pt", tokenize=False, add_generation_prompt=True)
        return output

    def generate(self, input: str, temperature: float=0.5, repetition_penalty: float=2.0, top_k: int=1, max_new_tokens: int=1024, early_stopping:bool=True, num_beams: int=1):
        st_time = time.time()
        
        # input = self.format_input(input)

        sequences = self.pipe(
            input,
            do_sample=True,
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )
        
        output = sequences[0]['generated_text']
        
        print(f"{output}")
        print(f"Time cost: {time.time()-st_time}s")

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and upload model to huggingface hub")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantized", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    generator = VietLLMGenerator(**vars(args))

    sample_input = "who is Ho Chi Minh"

    generator.generate(sample_input)
