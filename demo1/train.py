import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab


# 自己封装的评价指标
from metrics import (
    accuracy,
    precision,
    recall,
    f1
)


# Config
from config import parse_config


# Dataset
from dataset import (
    load_data,
    NewsDataset,
    get_collate_fn
)


# Model
from model import BertClassifier

# Seed
from utils import seed_everything

# Device

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# 评价函数

def evaluate(
    model,
    loader,
    num_classes
):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            mask = batch[
                "attention_mask"
            ].to(device)

            y = batch[
                "labels"
            ].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=mask
            )

            pred = torch.argmax(
                logits,
                dim=1
            )

            preds.extend(
                pred.cpu().tolist()
            )

            labels.extend(
                y.cpu().tolist()
            )


    # Accuracy
    acc = accuracy(
        preds,
        labels
    )


    # Precision
    p = precision(
        preds,
        labels,
        num_classes
    )


    # Recall
    r = recall(
        preds,
        labels,
        num_classes
    )


    # Macro-F1
    score_f1 = f1(
        preds,
        labels,
        num_classes
    )


    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": score_f1
    }

# 训练函数
def train(config):

    seed_everything()


    # 创建模型保存目录
    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    # SwanLab
    swanlab.init(

        project=config.project,

        experiment_name=config.experiment_name,

        config={
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "epochs": config.epochs
        }
    )

    # 1. 加载数据
    train_data = load_data(
        config.train_path
    )

    print(
        "训练集前3条：",
        train_data[:3]
    )


    dev_data = load_data(
        config.dev_path
    )


    test_data = load_data(
        config.test_path
    )

    # 2. 获取类别
    label_names = set()

    for item in train_data:

        label_names.add(
            item["label_name"]
        )


    labels = sorted(
        list(label_names)
    )


    num_classes = len(labels)


    print(
        "类别数量：",
        num_classes
    )


    # label -> id
    label2id = {

        label: i

        for i, label in enumerate(labels)

    }


    # id -> label
    id2label = {

        i: label

        for label, i in label2id.items()

    }
    # 3. 添加数字标签
    for dataset in [
        train_data,
        dev_data,
        test_data
    ]:

        for item in dataset:

            item["label"] = label2id[
                item["label_name"]
            ]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    # 动态 Padding
    collate_fn = get_collate_fn(
        tokenizer
    )

    # 4. Dataset
    train_dataset = NewsDataset(
        train_data,
        tokenizer,
        config
    )


    dev_dataset = NewsDataset(
        dev_data,
        tokenizer,
        config
    )


    test_dataset = NewsDataset(
        test_data,
        tokenizer,
        config
    )

    # DataLoader
    train_loader = DataLoader(

        train_dataset,

        batch_size=config.batch_size,

        shuffle=True,

        collate_fn=collate_fn

    )


    dev_loader = DataLoader(

        dev_dataset,

        batch_size=config.batch_size,

        shuffle=False,

        collate_fn=collate_fn

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=config.batch_size,

        shuffle=False,

        collate_fn=collate_fn

    )

    # 5. 创建模型
    model = BertClassifier(

        config,

        num_classes

    )

    model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.lr

    )


    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()


    # 最佳F1
    best = 0


    # 6. 训练
    for epoch in range(
        config.epochs
    ):

        model.train()

        total_loss = 0

        correct = 0

        total = 0


        for batch in train_loader:

            optimizer.zero_grad()


            input_ids = batch[
                "input_ids"
            ].to(device)


            mask = batch[
                "attention_mask"
            ].to(device)


            y = batch[
                "labels"
            ].to(device)


            # 模型预测
            logits = model(

                input_ids=input_ids,

                attention_mask=mask

            )


            # Loss
            loss = loss_fn(

                logits,

                y

            )


            # 反向传播
            loss.backward()


            # 更新参数
            optimizer.step()


            total_loss += loss.item()


            # Train Accuracy
            pred = torch.argmax(

                logits,

                dim=1

            )


            correct += (

                pred == y

            ).sum().item()


            total += y.size(0)


        # Train指标
        train_loss = (

            total_loss /

            len(train_loader)

        )


        train_accuracy = (

            correct /

            total

        )

        # 7. Dev验证
        result = evaluate(

            model,

            dev_loader,

            num_classes

        )


        print(
            "===================="
        )


        print(
            "Epoch:",
            epoch + 1
        )


        print(
            "Train Loss:",
            train_loss
        )


        print(
            "Train Accuracy:",
            train_accuracy
        )


        print(
            "Dev Accuracy:",
            result["accuracy"]
        )


        print(
            "Dev Precision:",
            result["precision"]
        )


        print(
            "Dev Recall:",
            result["recall"]
        )


        print(
            "Dev F1:",
            result["f1"]
        )


        # SwanLab
        swanlab.log({

            "Train Loss":
                train_loss,

            "Train Accuracy":
                train_accuracy,

            "Dev Accuracy":
                result["accuracy"],

            "Dev Precision":
                result["precision"],

            "Dev Recall":
                result["recall"],

            "Dev F1":
                result["f1"]

        })


        # 保存最佳模型
        if result["f1"] > best:

            best = result["f1"]


            torch.save(

                model.state_dict(),

                "checkpoints/best_model.pth"

            )


            print(
                "保存最佳模型，Dev F1 =",
                best
            )

    # 8. 加载最佳模型
    model.load_state_dict(

        torch.load(

            "checkpoints/best_model.pth",

            map_location=device

        )

    )


    # 9. Test Evaluation
    test_result = evaluate(

        model,

        test_loader,

        num_classes

    )


    print(
        "===================="
    )


    print(
        "Test Result"
    )


    print(
        "test accuracy:",
        test_result["accuracy"]
    )


    print(
        "test precision:",
        test_result["precision"]
    )


    print(
        "test recall:",
        test_result["recall"]
    )


    print(
        "test f1:",
        test_result["f1"]
    )

    # SwanLab记录Test
    swanlab.log({

        "Test Accuracy":
            test_result["accuracy"],

        "Test Precision":
            test_result["precision"],

        "Test Recall":
            test_result["recall"],

        "Test F1":
            test_result["f1"]

    })


# 主程序
if __name__ == "__main__":

    # 解析命令行参数
    config = parse_config()

    # 将config传入训练函数
    train(config)