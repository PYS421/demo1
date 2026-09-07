import torch
from torch import nn
from transformers import AutoModel


# 创建一个BERT文本分类器
class BertClassifier(nn.Module):

    def __init__(
        self,
        config,
        num_labels
    ):
        super().__init__()

        # BERT编码器
        self.bert = AutoModel.from_pretrained(
            config.model_name
        )

        # 防止过拟合
        self.dropout = nn.Dropout(
            0.1
        )

        # 分类层
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS向量
        cls = output.last_hidden_state[:, 0, :]

        # Dropout
        cls = self.dropout(cls)

        # 分类
        logits = self.classifier(cls)

        return logits