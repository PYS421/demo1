import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


def load_data(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = line.split("_!_")

            item = {
                "news_id": row[0],
                "label_code": row[1],
                "label_name": row[2],
                "title": row[3],
                "keywords": row[4]
            }

            data.append(item)

    return data


class NewsDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        config
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):

        return len(self.data)

    def __getitem__(
        self,
        index
    ):

        item = self.data[index]

        # 新闻标题
        text = str(
            item["title"]
        )

        # 标签
        label = item["label"]

        # Tokenizer
        encode = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True
        )

        # 添加标签
        encode["labels"] = label

        return encode


# 动态 Padding
def get_collate_fn(tokenizer):

    return DataCollatorWithPadding(
        tokenizer=tokenizer
    )