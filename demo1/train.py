import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab

# 自己封装的评价指标
from metrics import ClassificationMetrics

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
    loss_fn,
    num_classes
):

    model.eval()

    preds = []
    labels = []

    total_loss = 0

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

            total_loss += loss.item()

            # 获取预测类别
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

    # 创建指标对象
    metrics = ClassificationMetrics(
        preds,
        labels,
        num_classes
    )

    return {

        "loss":
            total_loss / len(loader),

        "accuracy":
            metrics.accuracy(),

        "precision":
            metrics.precision(),

        "recall":
            metrics.recall(),

        "f1":
            metrics.f1()
    }

# 训练函数
def train(config):

    # 设置随机种子
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

            "model_name":
                config.model_name,

            "max_length":
                config.max_length,

            "batch_size":
                config.batch_size,

            "lr":
                config.lr,

            "epochs":
                config.epochs
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


    # 保存最佳 Dev F1
    best = 0
    # 6. Train + Dev
    for epoch in range(

        config.epochs

    ):

        # Train
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


            # 计算 Loss
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


        # Train Loss
        train_loss = (

            total_loss /

            len(train_loader)

        )


        # Train Accuracy
        train_accuracy = (

            correct /

            total

        )

        # Dev
        result = evaluate(

            model,

            dev_loader,

            loss_fn,

            num_classes

        )


        # 打印结果
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
            "Dev Loss:",
            result["loss"]
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

            "Dev Loss":
                result["loss"],

            "Dev Accuracy":
                result["accuracy"],

            "Dev Precision":
                result["precision"],

            "Dev Recall":
                result["recall"],

            "Dev F1":
                result["f1"]

        })


        # 保存最佳 Dev Model

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


    # 训练结束
    print(
        "===================="
    )

    print(
        "训练完成"
    )

    print(
        "最佳 Dev F1:",
        best
    )


    # 7. 加载最佳 Dev Model
    model.load_state_dict(

        torch.load(

            "checkpoints/best_model.pth",

            map_location=device

        )

    )


    print(
        "已加载最佳 Dev Model"
    )


    # 8. Test
    test_result = evaluate(

        model,

        test_loader,

        loss_fn,

        num_classes

    )

    # Test结果

    print(
        "===================="
    )

    print(
        "Test Result"
    )

    print(
        "Test Loss:",
        test_result["loss"]
    )

    print(
        "Test Accuracy:",
        test_result["accuracy"]
    )

    print(
        "Test Precision:",
        test_result["precision"]
    )

    print(
        "Test Recall:",
        test_result["recall"]
    )

    print(
        "Test F1:",
        test_result["f1"]
    )


    # SwanLab记录Test

    swanlab.log({

        "Test Loss":
            test_result["loss"],

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