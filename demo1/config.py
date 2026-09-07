import argparse


class Config:

    def __init__(self):

        # 数据参数
        self.train_path = "D:/data/0.demo1文本分类/train_3k.txt"

        self.dev_path = "D:/data/0.demo1文本分类/dev_1k.txt"

        self.test_path = "D:/data/0.demo1文本分类/test_1k.txt"


        # 模型参数
        self.model_name = "google-bert/bert-base-chinese"

        self.max_length = 128

        # 训练参数
        self.batch_size = 16

        self.lr = 1e-5

        self.epochs = 10

        # SwanLab
        self.project = "demo1"

        self.experiment_name = "bert-text-classification"


def parse_config():

    parser = argparse.ArgumentParser(
        description="BERT文本分类训练参数"
    )

    # 数据参数
    parser.add_argument(
        "--train_path",
        type=str,
        default="D:/data/0.demo1文本分类/train_3k.txt",
        help="训练集路径"
    )

    parser.add_argument(
        "--dev_path",
        type=str,
        default="D:/data/0.demo1文本分类/dev_1k.txt",
        help="验证集路径"
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="D:/data/0.demo1文本分类/test_1k.txt",
        help="测试集路径"
    )

    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-chinese",
        help="预训练模型名称"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大文本长度"
    )

    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch Size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="学习率"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数"
    )

    # SwanLab参数
    parser.add_argument(
        "--project",
        type=str,
        default="demo1",
        help="SwanLab项目名称"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="bert-text-classification",
        help="实验名称"
    )


    # 解析参数
    args = parser.parse_args()

    return args