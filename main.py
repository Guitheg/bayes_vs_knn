import numpy as np

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import matplotlib.pyplot as plt

def separer(data,labels,pourcentage):
    nb_iter = int(pourcentage/100 * len(data))
    test_data = []
    test_labels = []
    for i in range(nb_iter):
        j = int(np.random.random_sample()*len(data))
        test_data.append(data[j])
        test_labels.append(data[j])
        data.pop(j)
        labels.pop(j)
    return data,labels,test_data,test_labels

def main():

    path = ["./tp6/data/data2.csv","./tp6/data/data3.csv","./tp6/data/data12.csv"]
    t = 321
    for p in path:
        data, labels = load_dataset(p)

        test_size = 10
        score_baye = []
        score_k_voisin = []
        x = []
        for i in range(0,85,5):
            a,b = baye_voisin(data,labels,test_size+i)
            score_baye += [a]
            score_k_voisin += [b]
            x += [test_size + i]
        print("t:",t)
        plt.subplot(t)
        plt.plot(x,score_baye,c='red')
        t +=  1
        print("t",t)
        plt.subplot(t)
        plt.plot(x,score_k_voisin,c='blue')
        t += 1

    plt.show()
    input("prout")


def baye_voisin(data, labels, ts):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = ts/100,random_state = 42)

    #GAUSSIENNE
    # Instanciation de la classe GaussianB
    g = GaussianBayes()
    # Apprentissage
    g.fit(train_data, train_labels)
    #g.graph(train_data, train_labels)
    # Score
    score_baye = g.score(test_data, test_labels)
    #K voisin
    n_neighbors = 10
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(train_data,train_labels)
    #Z = clf.predict(test_data)
    score_voisin = clf.score(test_data,test_labels)
    #print("test_size: ",ts,"score baye: ",score_baye,"score voisin: ",score_voisin)

    return score_baye,score_voisin


if __name__ == "__main__":
    main()
