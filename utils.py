import json
import pandas as pd
import os
from typing import List, Dict
from transformers import PretrainedModel


def read_json(fpath: str) -> Dict | List:
    pass


def read_text(fpath: str) -> str:
    pass


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pass


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pass


def write_csv(obj, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(fpath: str) -> PretrainedModel:
    pass
