import numpy as np
import os
from os.path import join
from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MAIN = os.path.abspath(os.path.dirname(__file__))
DATA = join(MAIN, "Data")

def main():
    path = [join(DATA, "data2.csv"), join(DATA, "data3.csv"), join(DATA, "data12.csv")]
    t = 321
    for p in path:
        data, labels = load_dataset(p)
        test_size = 10
        score_baye = []
        score_k_voisin = []
        x = []
        for i in range(0, 85, 5):
            s_bayes, cfx_bayes, s_knn, cfx_knn = baye_voisin(data, labels, test_size+i)
            score_baye += [s_bayes]
            score_k_voisin += [s_knn]
            x += [test_size + i]
        plt.subplot(t)
        plt.plot(x, score_baye, c='red')
        t += 1
        plt.subplot(t)
        plt.plot(x, score_k_voisin, c='blue')
        t += 1
    plt.show()

def baye_voisin(data, labels, ts):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = ts / 100, random_state = 42)

    # GAUSSIENNE
    g = GaussianBayes()
    g.fit(train_data, train_labels)

    # - Score:
    score_baye = g.score(test_data, test_labels)
    Z = g.predict(test_data)
    cfmat_bayes = confusion_matrix(test_labels, Z, labels=np.unique(test_labels))

    # K-NN
    n_neighbors = 10
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(train_data, train_labels)

    # - Score:
    score_voisin = clf.score(test_data, test_labels)
    Z = clf.predict(test_data)
    cfmat_knn = confusion_matrix(test_labels, Z, labels=np.unique(test_labels))
    
    return score_baye, cfmat_bayes, score_voisin, cfmat_knn

if __name__ == "__main__":
    main()
