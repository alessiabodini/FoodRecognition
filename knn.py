import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def knn(train_set, test_set, train_labels, test_labels):
    
    # 0. Apply a simple scaling on the data 
    ss = StandardScaler()
    train_set = ss.fit_transform(train_set)
    test_set = ss.transform(test_set)

    # 1. Set the parameters: K is the number of clostes neighbours to consider and fun is the metric chosen
    K = 1 # 1 or 3 or 5 or 7
    fun = 'cosine' # euclidean or correlation or minkowski or cosine
    n_classes = 10

    #print(train_set.shape)
    #print(test_set.shape)
    #print(train_labels.shape)
    #print(test_labels.shape)

    '''
    # 2.0. Direct method with KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=K, metric=fun)
    classifier.fit(train_set, train_labels)
    prediction = classifier.predict(test_set)
    '''

    # 2.1 Calculate the distance between train objects and test objects 
    D = cdist(train_set, test_set, metric=fun)
    #print(D.shape)

    # 3. For every test line (axis=0), order distances from smallest to largest 
    #    and find the K indexes of the train "points" closer
    k_neighbors = np.argsort(D, axis=0)[:K,:]
    #print(k_neighbors.shape)
    #print(k_neighbors)

    # 4. Check the labels of the K closest neighbours to find the most frequent using mode
    neighbors_labels = train_labels[k_neighbors]
    prediction = stats.mode(neighbors_labels, axis=0)[0][0]
    print('Finded the closest labels.')

    # 5. Calculate accurancy, precisione and recall
    '''
    for i in range(len(test_labels)):
        if prediction[i] == test_labels[i]:
            print(str(prediction[i]) + ' ---> n.' + str(i))
    '''
    accurancy = np.sum(prediction == test_labels) / len(test_labels)

    cmc = np.zeros((n_classes, n_classes))
    for predict, test_label in zip(prediction, test_labels):
        cmc[int(test_label), int(predict)] += 1.0
    
    precision = []
    recall = []
    for i in range(n_classes):
        if cmc[i,i] != 0:
            precision.append(cmc[i,i] / np.sum(cmc[:,i]))
            recall.append(cmc[i,i] / np.sum(cmc[i,:]))

    precision = np.mean(np.asarray(precision))
    recall = np.mean(np.asarray(recall))
    
    print('Classifier\'s accurancy: ' + '{0:.2f}'.format(accurancy * 100) + '%')
    print('Classifier\'s mean precision: ' + "{0:.2f}".format(precision))
    print('Classifier\'s mean recall : ' + "{0:.2f}".format(recall))

    #print(confusion_matrix(prediction, test_labels))
    #print(classification_report(prediction, test_labels))