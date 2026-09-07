def accuracy(
        preds,
        labels
):

    correct = 0

    for p, y in zip(preds, labels):

        if p == y:
            correct += 1

    return correct / len(labels)


def precision(
        preds,
        labels,
        num_classes
):

    result = []

    for cls in range(num_classes):

        pred_set = {
            i
            for i, p in enumerate(preds)
            if p == cls
        }

        true_set = {
            i
            for i, y in enumerate(labels)
            if y == cls
        }

        TP = len(
            pred_set & true_set
        )

        FP = len(
            pred_set - true_set
        )

        if TP + FP == 0:

            p = 0

        else:

            p = TP / (TP + FP)

        result.append(p)

    return sum(result) / num_classes


def recall(
        preds,
        labels,
        num_classes
):

    result = []

    for cls in range(num_classes):

        pred_set = {
            i
            for i, p in enumerate(preds)
            if p == cls
        }

        true_set = {
            i
            for i, y in enumerate(labels)
            if y == cls
        }

        TP = len(
            pred_set & true_set
        )

        FN = len(
            true_set - pred_set
        )

        if TP + FN == 0:

            r = 0

        else:

            r = TP / (TP + FN)

        result.append(r)

    return sum(result) / num_classes


def f1(
        preds,
        labels,
        num_classes
):

    result = []

    for cls in range(num_classes):

        pred_set = {
            i
            for i, p in enumerate(preds)
            if p == cls
        }

        true_set = {
            i
            for i, y in enumerate(labels)
            if y == cls
        }

        TP = len(
            pred_set & true_set
        )

        FP = len(
            pred_set - true_set
        )

        FN = len(
            true_set - pred_set
        )

        if TP + FP == 0:

            p = 0

        else:

            p = TP / (TP + FP)


        if TP + FN == 0:

            r = 0

        else:

            r = TP / (TP + FN)


        if p + r == 0:

            score = 0

        else:

            score = 2 * p * r / (p + r)


        result.append(score)


    return sum(result) / num_classes