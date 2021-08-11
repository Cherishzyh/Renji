import numpy as np
from sklearn.metrics import confusion_matrix


def Unsure(y_true, y_pred, f1_type='normal'):
    assert f1_type in ['normal', 'macro', 'micro']
    cm = confusion_matrix(y_true, y_pred)
    categories = cm.shape[0]
    precision = []
    recall = []
    f1_score = []
    if f1_type == 'normal':
        for cat in range(categories):
            P = cm[cat, cat] / sum(cm[cat, :])
            R = cm[cat, cat] / sum(cm[:, cat])
            precision.append(P)
            recall.append(R)
            f1_score.append(2 * P * R / (P + R))
    elif f1_type == 'micro':
        pass
    elif f1_type == 'macro':
        pass
    return precision, recall, f1_score


y_true = np.array([1, 2, 0, 3, 2, 1, 0, 2, 1, 2])
y_pred = np.array([1, 1, 2, 3, 2, 0, 1, 3, 1, 2])

p, r, fl = Unsure(y_true, y_pred, f1_type='normal')


