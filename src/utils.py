
from fastapi import UploadFile
import os, numpy as np, torch, pickle, random, math, json
from tempfile import NamedTemporaryFile
import cn2an
import re

# --- Utility Functions ---
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_temp_file(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name

def convert_to_chinese_readable(text: str) -> str:
    # Convert numbers to Chinese
    def replace_numbers(match):
        try:
            return cn2an.transform(match.group(), "an2cn")
        except:
            return match.group()
    text = re.sub(r'\d+(\.\d+)?', replace_numbers, text)

    # Convert symbols to Chinese equivalents
    symbol_map = {
        ".": "点",
        ",": "逗号",
        ":": "冒号",
        ";": "分号",
        "%": "百分之",
        "+": "加",
        "-": "减",
        "*": "乘",
        "/": "除",
        "=": "等于",
        "$": "美元",
        "(": "左括号",
        ")": "右括号"
    }
    for sym, zh in symbol_map.items():
        text = text.replace(sym, zh)

    return text
