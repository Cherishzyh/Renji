import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ShowCM(cm):
    sns.set()
    f, ax = plt.subplots()

    sns.heatmap(cm, annot=True, ax=ax) #画热力图

    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴

if __name__ == '__main__':
    cm = np.array()