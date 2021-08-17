from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
import seaborn as sns
from pylab import *


color_patten = sns.color_palette()


def ROC(label, pred, save_path=r''):

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(label, pred)
    auc = roc_auc_score(label, pred)
    plt.plot(fpn, sen, label='AUC: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    plt.close()