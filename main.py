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
    #path = [join(DATA, "data2.csv"), join(DATA, "data3.csv"), join(DATA, "data12.csv")]
    path = ["data2.csv","data3.csv","data12.csv"]
    #subplot init
    t = 321
    #Parcours de tout les fichiers tests.
    """
    for p in path:
        data, labels = load_dataset(join(DATA,p))
        train_size = 20
        score_baye = []
        score_k_voisin = []
        x = []
        for i in range(0, 80, 2):
            s_bayes, cfx_bayes, s_knn, cfx_knn = baye_voisin(data, labels, train_size+i)
            score_baye += [1-s_bayes]
            score_k_voisin += [1-s_knn]
            x += [train_size + i]
            #print(cfx_bayes)
            #print(cfx_knn)
        plt.subplot(t)
        plt.plot(x, score_baye, c='red')
        plt.title(p)
        plt.xlabel("Pourcentage du jeu d'apprentissage")
        plt.ylabel("Taux d'echec")
        t += 1
        plt.subplot(t)
        plt.plot(x, score_k_voisin, c='blue')
        plt.title(p)
        plt.xlabel("Pourcentage du jeu d'apprentissage")
        plt.ylabel("Taux d'echec")
        t += 1
    plt.show()
    """
    
    #affichage des matrices de confusions avec 80% d'apprentissage et 20% de test
    for p in path:
        data, labels = load_dataset(join(DATA,p))
        train_size = 10
        s_bayes, cfx_bayes, s_knn, cfx_knn = baye_voisin(data, labels, train_size)
        #print(cfx_bayes)
        #print(cfx_knn)
        plt.subplot(t)
        #plt.matshow(cfx_bayes,fignum=0)
        plt.imshow(cfx_bayes)

        plt.title(p)
        #plt.xlabel("Pourcentage du jeu d'apprentissage")
        #plt.ylabel("Taux d'echec")
        t += 1
        plt.subplot(t)
        #plt.matshow(cfx_knn,fignum=0)
        plt.imshow(cfx_knn,)
        plt.title(p)
        #plt.xlabel("Pourcentage du jeu d'apprentissage")
        #plt.ylabel("Taux d'echec")
        t += 1
    plt.show()

def baye_voisin(data, labels, ts):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size = ts / 100, random_state = 42)
    print("test taille:",len(test_data)," train test:",len(train_data)," test size: ",ts)
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
