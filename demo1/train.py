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


from config import *


# 添加 get_collate_fn
from dataset import (
    load_data,
    NewsDataset,
    get_collate_fn
)


from model import BertClassifier


from utils import seed_everything


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


            input_ids = batch["input_ids"].to(device)

            mask = batch["attention_mask"].to(device)

            y = batch["labels"].to(device)


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


    # 调用自己的 metrics.py
    acc = accuracy(

        preds,

        labels

    )


    p = precision(

        preds,

        labels,

        num_classes

    )


    r = recall(

        preds,

        labels,

        num_classes

    )


    # Macro-F1
    # 每个类别分别计算 F1
    # 最后再求平均
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
def train():


    seed_everything()


    # 创建模型保存目录
    os.makedirs(

        "checkpoints",

        exist_ok=True

    )


    # SwanLab
    swanlab.init(

        project="demo1",

        experiment_name="bert-text-classification"

    )


    # 1. 加载数据
    train_data = load_data(

        TRAIN_PATH

    )

    print(train_data[:3])


    dev_data = load_data(

        DEV_PATH

    )


    test_data = load_data(

        TEST_PATH

    )


    # 2. 获取类别
    # list + dict 写法
    label_names = set()


    for item in train_data:

        label_names.add(

            item["label_name"]

        )


    labels = sorted(

        list(label_names)

    )


    num_classes = len(labels)


    label2id = {


        label: i


        for i, label in enumerate(labels)


    }


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

        MODEL_NAME

    )



    # 动态 Padding
    collate_fn = get_collate_fn(

        tokenizer

    )


    # 4. Dataset / DataLoader
    train_loader = DataLoader(

        NewsDataset(

            train_data,

            tokenizer

        ),

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_fn

    )


    dev_loader = DataLoader(

        NewsDataset(

            dev_data,

            tokenizer

        ),

        batch_size=BATCH_SIZE,

        collate_fn=collate_fn

    )


    test_loader = DataLoader(

        NewsDataset(

            test_data,

            tokenizer

        ),

        batch_size=BATCH_SIZE,

        collate_fn=collate_fn

    )


    # 5. 创建模型
    model = BertClassifier(

        MODEL_NAME,

        num_classes

    )


    model.to(device)


    # 优化器
    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR

    )


    # 自己计算 Loss
    loss_fn = torch.nn.CrossEntropyLoss()


    best = 0


    # 6. 训练
    for epoch in range(EPOCHS):


        model.train()


        total_loss = 0


        for batch in train_loader:


            optimizer.zero_grad()


            y = batch["labels"].to(device)


            logits = model(

                input_ids=batch["input_ids"].to(device),

                attention_mask=batch["attention_mask"].to(device)

            )


            loss = loss_fn(

                logits,

                y

            )


            loss.backward()


            optimizer.step()


            total_loss += loss.item()


        # 7. 验证
        result = evaluate(

            model,

            dev_loader,

            num_classes

        )


        epoch_loss = (

            total_loss /

            len(train_loader)

        )


        print("====================")


        print(

            "epoch:",

            epoch + 1

        )


        print(

            "loss:",

            epoch_loss

        )


        print(

            "accuracy:",

            result["accuracy"]

        )


        print(

            "precision:",

            result["precision"]

        )


        print(

            "recall:",

            result["recall"]

        )


        print(

            "f1:",

            result["f1"]

        )


        # SwanLab记录
        swanlab.log({

            "epoch": epoch + 1,

            "loss": epoch_loss,

            "accuracy": result["accuracy"],

            "precision": result["precision"],

            "recall": result["recall"],

            "f1": result["f1"]

        })

        # 保存最佳模型
        if result["f1"] > best:


            best = result["f1"]


            torch.save(

                model.state_dict(),

                "checkpoints/best_model.pth"

            )


            print(

                "保存最佳模型"

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


    print("====================")

    print("Test Result")


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


    # SwanLab记录Test结果

    swanlab.log({

        "test_accuracy":

            test_result["accuracy"],

        "test_precision":

            test_result["precision"],

        "test_recall":

            test_result["recall"],

        "test_f1":

            test_result["f1"]

    })


# 主程序

if __name__ == "__main__":

    train()