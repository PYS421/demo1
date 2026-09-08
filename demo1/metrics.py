class ClassificationMetrics:

    def __init__(self, preds, labels, num_classes):
        self.preds = preds
        self.labels = labels
        self.num_classes = num_classes

        # 每个类别的 TP、FP、FN
        self.tp = [0] * num_classes
        self.fp = [0] * num_classes
        self.fn = [0] * num_classes

        # 只计算一次
        for p, y in zip(self.preds, self.labels):
            if p == y:
                self.tp[y] += 1
            else:
                self.fp[p] += 1
                self.fn[y] += 1

    def accuracy(self):
        correct = sum(
            p == y
            for p, y in zip(self.preds, self.labels)
        )

        return correct / len(self.labels)

    def precision(self):
        result = []

        for cls in range(self.num_classes):
            tp = self.tp[cls]
            fp = self.fp[cls]

            if tp + fp == 0:
                score = 0
            else:
                score = tp / (tp + fp)

            result.append(score)

        return sum(result) / self.num_classes

    def recall(self):
        result = []

        for cls in range(self.num_classes):
            tp = self.tp[cls]
            fn = self.fn[cls]

            if tp + fn == 0:
                score = 0
            else:
                score = tp / (tp + fn)

            result.append(score)

        return sum(result) / self.num_classes

    def f1(self):
        result = []

        for cls in range(self.num_classes):
            tp = self.tp[cls]
            fp = self.fp[cls]
            fn = self.fn[cls]

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                score = 0
            else:
                score = (
                    2 * precision * recall
                    / (precision + recall)
                )

            result.append(score)

        return sum(result) / self.num_classes