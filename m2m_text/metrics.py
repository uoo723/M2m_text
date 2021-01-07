"""
Created on 2021/01/05
@author Sangwoo Han
"""


def get_accuracy(labels, targets, num_classes):
    correct_mask = labels == targets
    correct_total = len(targets)
    correct = correct_mask.sum()

    # major_mask = targets < num_classes // 3
    # major_total = major_mask.sum()
    # major_correct = (correct_mask * major_mask).sum()

    # minor_mask = targets >= num_classes - num_classes // 3
    # minor_total = minor_mask.sum()
    # minor_correct = (correct_mask * minor_mask).sum()

    # neutral_mask = ~(major_mask + minor_mask)
    # neutral_total = neutral_mask.sum()
    # neutral_correct = (correct_mask * neutral_mask).sum()

    # class_correct = np.zeros(num_classes)
    # class_total = np.zeros(num_classes)

    # for i in range(num_classes):
    #     class_mask = targets == i
    #     class_total[i] = class_mask.sum()
    #     class_correct[i] = (correct_mask * class_mask).sum()

    return {
        "acc": correct / correct_total,
        # "major_acc": major_correct / major_total,
        # "minor_acc": minor_correct / minor_total,
        # "neutral_acc": neutral_correct / neutral_total,
        # "class_acc": class_correct / class_total,
    }
