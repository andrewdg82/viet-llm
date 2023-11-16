import argparse
import torch
import wandb
from loguru import logger
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

def prepair_input_text(input: str, target: str, context: str = "") -> str:
    text = "<|system|>\nYou are a support chatbot who helps with user queries chatbot who always responds in the style of a professional."
    if context is not None and context != "":
        text += "\nContext\n" + context
    text += "\n<|user|>\n" + input 
    text += "\n<|assistant|>\n" + target
    return text

def prepair_input_text2(input: str, target: str, context: str = "") -> str:
    text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    if context is not None and context != "":
        text += "\n### Instruction:\n" + context
    text += "\n### Input:\n" + input
    text += "\n### Response:\n" + target
    return text

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = text_formatting(example['input'][i], example['target'][i], example['context'][i] if "context" in example else None)
        output_texts.append(text)
    return output_texts

def text_formatting(data):
    if data['context']:
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["input"]} \n\n### Input:\n{data["context"]}\n\n### Response:\n{data["target"]}"""
    else:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["input"]}\n\n### Response:\n{data["target"]}""" 
    return text


class VietLLMTrainer:
    def __init__(self, args: argparse.Namespace):

        data_files = args.data.split(",")
        datasets = []
        for data_file in data_files:
            dataset = load_dataset("csv", data_files=data_file, split="train", num_proc=8)
            print("dataset:", dataset)
            datasets.append(dataset)
        
        train_dataset = concatenate_datasets(datasets)
        print(train_dataset)
    
       # Load base model(Mistral 7B)
        bnb_config = BitsAndBytesConfig(  
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
                load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
        )
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        tokenizer.add_bos_token, tokenizer.add_eos_token

        print(model)

        #Adding the adapters in the layers
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
        )
        model = get_peft_model(model, peft_config)

        #Hyperparamter
        training_arguments = TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.acc_step,
            optim=args.optimizer,
            learning_rate=args.lr,
            lr_scheduler_type=args.scheduler,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_steps=args.logging_step,
            num_train_epochs=args.num_epoch,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            report_to="wandb",
            push_to_hub=False
        )


        # Setting sft parameters
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            peft_config=peft_config,
            max_seq_length=None,
            tokenizer=tokenizer,
            args=training_arguments,
            packing= False,
        )

        logger.info(f"Init LLM Trainer done")

    def run(self):
        logger.info(f"Start finetuning...")
        wandb.login()
        run = wandb.init(
            project='Fine tuning mistral 7B', 
            job_type="training", 
            anonymous="allow"
        )
        self.trainer.train()
        logger.success(f"Completed")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Finetuning args")

    parser.add_argument("-m", "--model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B") # HuggingFaceH4/zephyr-7b-alpha
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--num_epoch", type=int, default=2)
    parser.add_argument("-a", "--acc_step", type=int, default=1)
    parser.add_argument("-op", "--optimizer", type=str, default="paged_adamw_32bit")
    parser.add_argument("-o", "--output", type=str, default="output", help="Specify the folder output")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scheduler", type=str, default="cosine") # constant
    parser.add_argument("--data", type=str, default="data/data.json", help="Specify data")
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    args = parser.parse_args()
    logger.info(args)

    trainer = VietLLMTrainer(args)
    trainer.run()
