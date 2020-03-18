import numpy as np
import os
from os.path import join
from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MAIN = os.path.abspath(os.path.dirname(__file__))
DATA = join(MAIN, "Data")

def main():
    #path = [join(DATA, "data2.csv"), join(DATA, "data3.csv"), join(DATA, "data12.csv")]
    path = ["data2.csv","data3.csv","data12.csv"]
    #matrice_confu(path,20)
    taux_erreur(path)
    #cross_validation(path)
    #data, labels = load_dataset(join(DATA, path[0]))
    #baye_voisin(data,labels,90)

#Ne marche pas, probleme dans les caluls de baye, je n'ai pas trouve de solution
def cross_validation(path):
    kf = KFold(n_splits=5)
    z = np.zeros(100)
    data, labels = load_dataset(join(DATA, path[0]))
    data = np.array(data)
    labels = np.array(labels)
    for train, test in kf.split(data):
        """
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for indice in train:
            train_data.append(data[indice])
            train_labels.append(labels[indice])
        for indice in test:
            test_data.append(data[indice])
            test_labels.append(labels[indice])
        print(train)
        print(test)

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)"""


        #train_data = data[0:800]
        #train_labels = labels[0:800]
        #test_data = data[801:999]
        #test_labels = labels[801:999]
        train_data = data[train]
        train_labels = labels[train]
        test_data = data[test]
        test_labels = labels[test]
        print(len(train_data))
        print(len(train_labels))
        print(len(test_data))
        print(len(test_labels))
        #print(train_data)
        #print(train_labels)
        #print(test_data)
        #print(test_labels)
        
        # GAUSSIENNE
        g = GaussianBayes()
        g.fit(train_data, train_labels)

        # - Score:
        score_baye = g.score(test_data, test_labels)

        # K-NN
        n_neighbors = 10
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
        clf.fit(train_data, train_labels)

        # - Score:
        score_voisin = clf.score(test_data, test_labels)

        print(score_voisin) 
        break       



#Parcours de tout les fichiers tests pour le calcul du taux d'erreur
def taux_erreur(path):
    #subplot init
    t = 321
    for p in path:
        data, labels = load_dataset(join(DATA,p))
        train_size = 20
        score_baye = []
        score_k_voisin = []
        x = []
        for i in range(0, 80, 10):
            s_bayes, cfx_bayes, s_knn, cfx_knn = baye_voisin(data, labels, train_size+i)
            score_baye += [1-s_bayes]
            score_k_voisin += [1-s_knn]
            x += [train_size + i]
            #print(cfx_bayes)
            #print(cfx_knn)
        #Baye
        plt.subplot(t)
        plt.plot(x, score_baye, c='red')
        plt.title(p)
        plt.xlabel("Pourcentage du jeu d'apprentissage")
        plt.ylabel("Taux d'echec")
        t += 1
        #Knn
        plt.subplot(t)
        plt.plot(x, score_k_voisin, c='blue')
        plt.title(p)
        plt.xlabel("Pourcentage du jeu d'apprentissage")
        plt.ylabel("Taux d'echec")
        t += 1
    plt.show()

    
#affichage des matrices de confusions avec 80% d'apprentissage et 20% de test
def matrice_confu(path,train_size):
    #subplot init
    t = 321
    for p in path:
        data, labels = load_dataset(join(DATA,p))
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
