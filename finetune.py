import argparse
import torch
from loguru import logger
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer

def prepair_input_text(input: str, target: str, context: str = "") -> str:
    text = "<|system|>\nYou are a support chatbot who helps with user queries chatbot who always responds in the style of a professional."
    if context != "":
        text += "\nContext\n" + context
    text += "\n<|user|>\n" + input 
    text += "\n<|assistant|>\n" + target
    return text

def prepair_input_text2(input: str, target: str, context: str = "") -> str:
    text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    text += "\n### Instruction:\n" + context
    text += "\n### Input:\n" + input
    text += "\n### Response:\n" + target

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = prepair_input_text2(example['question'][i], example['answer'][i], example['context'][i])
        output_texts.append(text)
    return output_texts

class VietLLMTrainer:
    def __init__(self, args: argparse.Namespace):

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = GPTQConfig(
            bits=4,
            disable_exllama=True,
            device_map="auto",
            use_cache=False,
            lora_r=16,
            lora_alpha=16,
            tokenizer=tokenizer
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,
        )

        model.config.use_cache=False
        model.config.pretraining_tp=1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )

        model = get_peft_model(model, peft_config)

        print(model)

        training_arguments = TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.acc_step,
            optim=args.optimizer,
            learning_rate=args.lr,
            lr_scheduler_type=args.scheduler,
            save_strategy="epoch",
            logging_steps=args.logging_step,
            num_train_epochs=args.num_epoch,
            fp16=True,
            push_to_hub=False
        )

        dataset = load_dataset("json", data_files=args.data)
        dataset = dataset["train"].train_test_split(test_size=0.2)
        
        dataset2 = load_dataset("vietgpt/mfag_vi")
        valid_ds2 = dataset2.pop("validation")
        dataset2["test"] = valid_ds2

        dataset2["train"] = dataset2["train"].add_column("context", ['']*len(dataset2["train"]))
        dataset2["test"] = dataset2["test"].add_column("context", ['']*len(dataset2["test"]))

        train_dataset = concatenate_datasets([dataset["train"], dataset2["train"]])
        valid_dataset = concatenate_datasets([dataset["test"], dataset2["test"]])


        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            peft_config=peft_config,
            # dataset_text_field="text",
            formatting_func=formatting_prompts_func,
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=args.max_seq_len
        )
        logger.info(f"Init LLM Trainer done")

    def run(self):
        logger.info(f"Start finetuning...")
        self.trainer.train()
        logger.success(f"Completed")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Finetuning args")

    parser.add_argument("-m", "--model", type=str, default="TheBloke/zephyr-7B-alpha-GPTQ") # HuggingFaceH4/zephyr-7b-alpha
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-e", "--num_epoch", type=int, default=2)
    parser.add_argument("-a", "--acc_step", type=int, default=1)
    parser.add_argument("-op", "--optimizer", type=str, default="paged_adamw_32bit")
    parser.add_argument("-o", "--output", type=str, default="output", help="Specify the folder output")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scheduler", type="str", default="cosine")
    parser.add_argument("--data", type=str, default="data/data.json", help="Specify data")
    parser.add_argument("--logging_step", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    args = parser.parse_args()
    logger.info(args)

    trainer = VietLLMTrainer(args)
    trainer.run()
