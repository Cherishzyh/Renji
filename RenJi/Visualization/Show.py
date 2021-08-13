import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ShowCM(cm):
    sns.set()
    f, ax = plt.subplots()

    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', cbar=False) #画热力图

    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()

if __name__ == '__main__':
    cm = np.array([[13, 5, 0, 1], [11, 6, 4, 2], [2, 6, 3, 0], [3, 0, 0, 15]])
    ShowCM(cm)