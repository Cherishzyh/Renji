import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def F1Score(y_true, y_pred, label=None):
    cm = confusion_matrix(y_true, y_pred, labels=label)
    categories = cm.shape[0]
    precision = []
    recall = []
    f1_score = []
    for cat in range(categories):
        P = cm[cat, cat] / sum(cm[cat, :])
        R = cm[cat, cat] / sum(cm[:, cat])
        precision.append(P)
        recall.append(R)
        if P == 0 and R == 0:
            f1_score.append(0.)
        else:
            f1_score.append(2 * P * R / (P + R))
    return precision, recall, f1_score, cm

def AUC(y_true, y_pred):
    fpn, sen, the = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return


    pass



if __name__ == "__main__":
    y_true = np.array([1, 2, 0, 3, 2, 1, 0, 2, 1, 2])
    y_pred = np.array([1, 1, 2, 3, 2, 0, 1, 3, 1, 2])

    p, r, fl, cm = F1Score(y_true, y_pred)
    print(p)


