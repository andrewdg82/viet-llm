import argparse
import torch
import wandb
from loguru import logger
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import glob
import os
import humanfriendly

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Model summary: Total params: {humanfriendly.format_number(all_param)} || Trainable params: {humanfriendly.format_number(trainable_params)} ({100 * trainable_params / all_param}%)")

def prepair_input_text(input: str, target: str, context: str = "") -> str:
    text = "<|system|>\nYou are a support chatbot who helps with user queries chatbot who always responds in the style of a professional."
    if context is not None and context != "":
        text += "\nContext\n" + context
    text += "\n<|user|>\n" + input 
    text += "\n<|assistant|>\n" + target
    return text

def text_formating(input: str, target: str, context: str = "") -> str:
    if context is not None and context != "":
        text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        text += "\n### Input:\n" + context
    else:
        text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    text += "\n### Instruction:\n" + input
    text += "\n\n### Response:\n" + target
    return text

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = text_formating(example['input'][i], example['target'][i], example['context'][i] if "context" in example else None)
        output_texts.append(text)
    return output_texts


class VietLLMTrainer:
    def __init__(self, args: argparse.Namespace):

        if os.path.isdir(args.data):
                data_files = glob.glob(args.data + "/*.csv")
        else:
            data_files = args.data.split(",")

        datasets = []

        for data_file in data_files:
            dataset = load_dataset("csv", data_files=data_file, split="train")
            datasets.append(dataset)
        
        train_dataset = concatenate_datasets(datasets)
        print(train_dataset)
        train_dataset = train_dataset.to_iterable_dataset()
        train_dataset = train_dataset.shuffle()
    
       # Load base model(Mistral 7B)
        bnb_config = BitsAndBytesConfig(  
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
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
        
        print_trainable_parameters(model)

        #Adding the adapters in the layers
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
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
            fp16=args.fp16,
            bf16=args.bf16,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=args.group_by_length,
            dataloader_num_workers=args.num_workers,
            report_to="wandb",
            push_to_hub=False
        )


        # Setting sft parameters
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            peft_config=peft_config,
            max_seq_length=args.max_seq_len,
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
            name=args.model,
            job_type="training", 
            anonymous="allow"
        )
        self.trainer.train()
        logger.success(f"Completed")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Finetuning args")

    parser.add_argument("-m", "--model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B") # HuggingFaceH4/zephyr-7b-alpha
    parser.add_argument("-d", "--data", type=str, default="data", help="Specify data")
    parser.add_argument("-o", "--output", type=str, default="models", help="Specify the folder output")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--num_epoch", type=int, default=2)
    parser.add_argument("-a", "--acc_step", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="paged_adamw_32bit")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="cosine") # constant
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=True)

    args = parser.parse_args()
    logger.info(args)

    trainer = VietLLMTrainer(args)
    trainer.run()
