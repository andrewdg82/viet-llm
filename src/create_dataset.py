from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import random
import numpy as np
import logging
import multiprocessing
from typing import List
from joblib import Parallel, delayed
from functools import reduce
import operator

def load_spoken_norm_data():
    print(f"Loading spoken_norm_assignment dataset")
    ds = load_dataset("VietAI/spoken_norm_assignment")

    questions = [
        "Chuẩn hóa câu sau từ dạng nói sang dạng viết: ",
        "Chuẩn hóa câu sau sang dạng viết: ",
        "Chuẩn hóa câu sau: ",
        "Normalize spoken form to written form for this text: ",
        "Normalize this text: ",
    ]

    questions2 = [
        "Chuẩn hóa câu sau từ dạng viết sang dạng nói: ",
        "Chuẩn hóa câu sau sang dạng nói: ",
        "Normalize written form to spoken form for this text: ",
        "Normalize this text to spoken form: ",
    ]

    def create_sample(item: dict):
        samples = []
        for spk_word, wrt_word in zip(item["src"], item["tgt"]):
            if spk_word != wrt_word:
                d = {
                    "input": random.choice(questions) + spk_word if np.random.rand() > 0.5 else spk_word,
                    "target": wrt_word
                }
                d2 = {
                    "input": random.choice(questions2) + wrt_word if np.random.rand() > 0.5 else wrt_word,
                    "target": spk_word
                }
                samples.append(d)
                samples.append(d2)

        return samples
    
    data = []
    for item in tqdm(ds["train"]):
        data += create_sample(item)

    ds = Dataset.from_list(data)
    return ds

def load_xp3_dataset():
    print(f"Loading bigscience/xP3 dataset")
    ds = load_dataset("bigscience/xP3", "vi")
    ds = ds["train"].rename_column("inputs", "input")
    ds = ds.rename_column("targets", "target")
    print(ds[0])
    return ds

def load_mfag_vi_dataset():
    print(f"Loading nlplabtdtu/mfag_vi dataset")
    ds = load_dataset("nlplabtdtu/mfag_vi")
    ds_train = ds["train"].rename_column("question", "input")
    ds_train = ds_train.rename_column("answer", "target")

    ds_valid = ds["validation"].rename_column("question", "input")
    ds_valid = ds_valid.rename_column("answer", "target")

    ds = concatenate_datasets([ds_train, ds_valid])
    print(ds[0])
    return ds

def load_news_corpus_dataset(percent: float):
    print(f"Loading news_corpus dataset")

    ds = load_dataset("hieunguyen1053/binhvq-news-corpus", split=f"train[{percent}%:{percent + 10}%]")
    summary_questions = [
        "Tóm tắt đoạn văn sau: ",
        "Tóm tắt đoạn sau: ",
        "Tóm tắt đoạn text: ",
        "Đoạn văn này nói về vấn đề gì: ",
        "Hãy tổng hợp đoạn văn sau thành vài câu ngắn gọn: ",
        "Tóm tắt: ",
        "Rút gọn đoạn text sau: ",
        "Viết tiêu đề cho đoạn sau: ",
        "Viết title cho đoạn text sau: "
    ]

    classify_question = [
        "Câu sau nói về chủ đề gì: ",
        "Đoạn văn sau nói về chủ đề gì: ",
        "Chỉ ra chủ đề của đoạn văn sau: ",
        "Gán nhãn chủ đề cho câu sau: ",
        "Lĩnh vực được đề cập đến của câu sau là gì: ",
        "Đoạn văn sau nói về lĩnh vực gì: "
    ]

    def create_sample(item: dict)->List[dict]:
        content = item["content"]
        content = content.split("\n")[:-3]
        item["content"] = "\n".join(content)
        sample_write_content = {
            "input": item["title"],
            "target": item["content"]
        }
        sample_summary = {
            "input": random.choice(summary_questions) + item["content"],
            "target": item["title"] if np.random.rand() > 0.5 else item["summary"]
        }
        sample_classify = {
            "input": random.choice(classify_question) + item["content"] if np.random.rand() > 0.5 else item["summary"],
            "target": item["category"]
        }
        sample_classify2 = {
            "input": random.choice(classify_question) + item["title"] if np.random.rand() > 0.5 else item["summary"],
            "target": item["category"]
        }
        return [sample_write_content, sample_summary, sample_classify, sample_classify2]
    
    data = []
    for item in tqdm(ds):
        data += create_sample(item)
    
    ds = Dataset.from_list(data)
    return ds

def load_MBZUAI_BactrianX_dataset():
    print(f"Loading MBZUAI_BactrianX dataset")
    ds = load_dataset("MBZUAI/Bactrian-X", "vi")
    ds = ds["train"].rename_column("output", "target")
    ds = ds.rename_column("input", "context")
    ds = ds.rename_column("instruction", "input")
    ds = ds.remove_columns("id")
    print(ds[0])
    return ds

def load_json(filepath: str="data/data_v1_426k.json"):
    print(f"Loading dataset from json file: {filepath}")
    ds = load_dataset("json", data_files=filepath)
    ds = ds["train"].rename_column("question", "input")
    ds = ds.rename_column("answer", "target")
    return ds

def clean_text(text: str) -> str:
    return text

def create_dataset(save_dir: str):
    list_dataset = [
        # load_spoken_norm_data,
        # load_xp3_dataset,
        # load_mfag_vi_dataset,
        # load_json,
        load_news_corpus_dataset,
        # load_MBZUAI_BactrianX_dataset
    ]

    for func in list_dataset:
        print(func.__name__)
        if func.__name__ == "load_news_corpus_dataset":
            for percent in range(30, 100, 10):
                ds = func(percent)
                ds = ds.filter(lambda item: item["input"] is not None and item["target"] is not None and len(item["input"].split()) > 3 and len(item["target"].split()) > 2)
                ds.to_csv(f"{save_dir}/{func.__name__}_{percent}.{percent+10}.csv")
                del ds
        else:
            ds = func()
            ds = ds.filter(lambda item: item["input"] is not None and item["target"] is not None)
            ds = ds.filter(lambda item: len(item["input"].split()) > 3 and len(item["target"].split()) > 2)
            ds.to_csv(f"{save_dir}/{func.__name__}.csv")


if __name__=="__main__":
    create_dataset("data")
