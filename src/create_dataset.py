from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import random
import numpy as np

def load_spoken_norm_data():
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

    return Dataset.from_list(data)

def load_xp3_dataset():
    ds = load_dataset("bigscience/xP3", "vi")
    ds = ds.rename_column("inputs", "input")
    ds = ds.rename_column("targets", "target")
    return ds

def load_mfag_vi_dataset():
    ds = load_dataset("nlplabtdtu/mfag_vi")

def load_json(filepath: str):
    ds = load_dataset("json", data_files=filepath)

def create_dataset(save_filepath: str):
    # ds = load_spoken_norm_data()
    # ds = load_xp3_dataset()
    ds = load_mfag_vi_dataset()

if __name__=="__main__":
    create_dataset("data/data.json")
