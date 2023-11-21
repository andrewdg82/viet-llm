import argparse
from typing import Dict
import torch
import wandb
import transformers
from loguru import logger
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from tqdm import tqdm
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
    print(f"Model summary: Total params: {humanfriendly.format_number(all_param)} || Trainable params: {humanfriendly.format_number(trainable_params)} ({100 * trainable_params / all_param}) %")        


class VietLLMTrainer:
    def __init__(self, args: argparse.Namespace):
        bnb_config = BitsAndBytesConfig(  
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print(model)
        print_trainable_parameters(model)
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()

        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
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
                "lm_head"
            ],
        )
        model = get_peft_model(model, peft_config)
        
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
            length_column_name="lengths",
            dataloader_num_workers=args.num_workers,
            report_to="wandb",
            push_to_hub=False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        tokenizer.add_bos_token, tokenizer.add_eos_token
        
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.peft_config = peft_config
        self.training_arguments = training_arguments
        
        logger.info(f"Init model done")

    def run(self):
        logger.info(f"Load dataset...")
        if os.path.exists(os.path.join(self.args.data, "processed")):
            logger.info(f"Load processed dataset from: {os.path.join(self.args.data, 'processed')}")
            dataset = load_from_disk(os.path.join(self.args.data, "processed"))
        else:
            dataset = self._load_dataset(self.args.data, use_iterable=False)
            print(dataset)

            logger.info(f"Preprocessing dataset...")
            dataset = preprocess(dataset, self.tokenizer)
            
            logger.info(f"Save dataset to: {os.path.join(self.args.data, 'processed')}")
            dataset.save_to_disk(os.path.join(self.args.data, "processed"), max_shard_size="10GB", num_proc=self.args.num_workers)

        print(dataset)
        
        logger.info(f"Filter long sample (length > {self.args.max_seq_len})")
        dataset = filter(dataset, self.args.max_seq_len)
        print(dataset)

        print(dataset[0])
        
        logger.info(f"Prepairing STFT Trainer...")
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=self.peft_config,
            max_seq_length=self.args.max_seq_len,
            tokenizer=self.tokenizer,
            dataset_num_proc=self.args.num_workers,
            args=self.training_arguments,
            dataset_text_field="text",
            packing=False,
        )

        logger.info(f"Init wandb...")
        wandb.login()
        wandb.init(
            project='Fine tuning OpenHermes-2.5-Mistral-7B',
            name=args.output.split("/")[-1],
            job_type="training", 
            anonymous="allow"
        )
        
        logger.info(f"Start finetuning...")
        trainer.train()

        logger.success(f"Completed")
    
    def _load_dataset(self, data_dir: str, use_iterable: bool=False):
        if os.path.isdir(data_dir):
            data_files = glob.glob(data_dir + "/*.csv")
        else:
            data_files = data_dir.split(",")

        datasets = []
        for data_file in data_files:
            # if "load_json" not in data_file: continue
            dataset = load_dataset("csv", data_files=data_file, split="train")
            datasets.append(dataset)
        
        dataset = concatenate_datasets(datasets)

        if use_iterable:
            dataset = dataset.to_iterable_dataset()
            dataset = dataset.shuffle()
        return dataset

def text_formating(sample: Dict[str, str], tokenizer: transformers.PreTrainedTokenizerBase, tokenize: bool=False) -> str:
    if "context" in sample and sample["context"] is not None:
        message = [
            {"role": "system", "content": "Bạn tên là Linh. Bạn là một cô trợ lý ảo được tạo ra bởi Techainer, bạn có nhiệm vụ trả lời các câu hỏi của con người một cách chính xác và tự nhiên nhất."},
            {"role": "user", "content": f"Dựa vào dữ liệu dưới đây trả lời đâu hỏi: {sample['input']}\n\nDữ liệu:\n{sample['context']}"},
            {"role": "assistant", "content": sample["target"]}
        ]
    else:
        message = [
            {"role": "system", "content": "Bạn tên là Linh. Bạn là một cô trợ lý ảo được tạo ra bởi Techainer, bạn có nhiệm vụ trả lời các câu hỏi của con người một cách chính xác và tự nhiên nhất."},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["target"]}
        ]
    output = tokenizer.apply_chat_template(message, return_tensors="pt", tokenize=tokenize)
    return output

def preprocess(dataset: Dataset, tokenizer: transformers.PreTrainedTokenizerBase) -> Dataset:
    dataset = dataset.map(
        lambda sample: {
            "text": text_formating(sample, tokenizer),
            "lengths": len(text_formating(sample, tokenizer, tokenize=True)[0])
            },
        remove_columns=["input", "target", "context"],
        num_proc=12
    )
    return dataset            

def filter(dataset: Dataset, max_seq_len: int) -> Dataset:
    dataset = dataset.filter(lambda example: example["lengths"] <= max_seq_len, num_proc=12)
    return dataset

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Finetuning args")

    parser.add_argument("-m", "--model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B") # HuggingFaceH4/zephyr-7b-alpha
    parser.add_argument("-d", "--data", type=str, default="data", help="Specify data")
    parser.add_argument("-o", "--output", type=str, default="models", help="Specify the folder output")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--num_epoch", type=int, default=1)
    parser.add_argument("-a", "--acc_step", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="paged_adamw_32bit")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="constant") # constant
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=True)

    args = parser.parse_args()
    logger.info(args)

    trainer = VietLLMTrainer(args)
    trainer.run()
