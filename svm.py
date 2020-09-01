import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def svm(train_set, test_set, train_labels, test_labels):

    print(train_set.shape)

    # 0. Apply a simple scaling on the data 
    ss = StandardScaler()
    train_set = ss.fit_transform(train_set)
    test_set = ss.transform(test_set)

    # 1. Initialize parameters 
    kernel = 'rbf' # linear or poly or rbf
    C = 1
    gamma = 'auto'
    max_iteration = 10 # 1000
    n_classes = 101

    # 2. Initialize a SVM classification model for every single class and train all of them
    #    (in this case only the first 11 class (with initial 'a' and 'b') are considered for semplicity)
    models = []
    print(train_set.shape)
    print(train_labels.shape)
    for i in range(n_classes):
        models.append(SVC(C=C, gamma=gamma, kernel=kernel, max_iter=max_iteration, probability=True))
        # use decision_function_shape = 'ovr' in SVC for the version 'one vs rest'
        models[i].fit(train_set, train_labels == i)
        print(i)
    print('\nFitting ended.')

    # 3. Classify testing set's data and build the confusione matrix
    predicted_scores = []
    for i in range(n_classes):
        predicted_scores.append(models[i].predict_proba(test_set)[:,1])
    print('Priting scores...')

    predicted_scores = np.asarray(predicted_scores)
    predicted = np.argmax(predicted_scores, axis=0)

    '''
    cmc = np.zeros((n_classes, n_classes))
    for predict, test_label in zip(predicted, test_labels):
        cmc[test_label, predict] += 1.0
    '''

    # 4. Make the confusion matrix and calculate the mean of accurancy, precision and recall,
    #    compared to all the classes 

    print('Printing accurancy...')
    cmc = confusion_matrix(predicted, test_labels)
    accurancy = np.sum(np.diagonal(cmc)) / np.sum(cmc)

    '''
    precision = []
    recall = []
    for i in range(n_classes):
        if cmc[i,i] != 0:
            precision.append(cmc[i,i] / np.sum(cmc[:,i]))
            recall.append(cmc[i,i] / np.sum(cmc[i,:]))

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    precision = np.mean(precision)
    recall = np.mean(recall)
    '''

    #print(cmc)
    print(classification_report(predicted, test_labels))
    print('Classifier\'s accurancy: ' + "{0:.2f}".format(accurancy*100) + '%')
    #print('Classifier\'s mean precision: ' + "{0:.2f}".format(precision))
    #print('Classifier\'s mean recall : ' + "{0:.2f}".format(recall))