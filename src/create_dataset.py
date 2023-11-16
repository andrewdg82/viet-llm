from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import random
import numpy as np
import logging

def load_spoken_norm_data():
    print(f"Loading spoken_norm_assignment dataset")
    data = []
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


    for item in tqdm(ds["train"]):
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
                data.append(d)
                data.append(d2)
    print(data[0])
    return Dataset.from_list(data)

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

def load_news_corpus_dataset():
    ds = load_dataset("hieunguyen1053/binhvq-news-corpus")
    print(ds)
    data = []
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

    for item in tqdm(ds["train"]):
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
        data += [sample_write_content, sample_summary, sample_classify, sample_classify2]

    ds = Dataset.from_list(data)
    return ds

def load_json(filepath: str):
    print(f"Loading dataset from json file: {filepath}")
    ds = load_dataset("json", data_files=filepath)
    ds = ds["train"].rename_column("question", "input")
    ds = ds.rename_column("answer", "target")
    ds = ds.filter(lambda item: len(item["input"].split()) > 3 and len(item["target"].split()) > 2)
    return ds

def clean_text(text: str) -> str:
    return text

def create_dataset(save_filepath: str):
    list_dataset = [
        load_spoken_norm_data(),
        load_xp3_dataset(),
        load_mfag_vi_dataset(),
        load_json("data/data_v1_426k.json"),
        load_news_corpus_dataset()
    ]
    print(list_dataset)
    concat_ds = concatenate_datasets(list_dataset)
    print(concat_ds)
    concat_ds = concat_ds.filter(lambda item: len(item["input"].split()) > 3 and len(item["target"].split()) > 2)
    print(concat_ds)

    concat_ds.to_csv(save_filepath)


if __name__=="__main__":
    create_dataset("data/data.csv")
