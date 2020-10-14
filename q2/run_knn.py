from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    class_list = []
    k_sets = [1,3,5,7,9]
    for k in k_sets:
        prediction = knn(k, train_inputs, train_targets, valid_inputs)
        counter = 0
        for i in range(prediction.size):
            if(prediction[i]==valid_targets[i]): counter += 1
        class_rate = counter / valid_targets.size
        class_list.append(class_rate)
        
    plt.plot(k_sets,class_list,label="valiadtion")
    plt.xlabel("k")
    plt.ylabel("classification_rate")
    plt.xticks(k_sets)
    plt.legend()
    plt.show()
    k_test = [1,3,5]

if __name__ == "__main__":
    run_knn()
    train_inputs, train_targets = load_train()
    test_inputs, test_targets = load_test()
    k_test = [3,5,7]
    class_list = []
    for k in k_test:
        prediction = knn(k, train_inputs, train_targets, test_inputs)
        counter = 0
        for i in range(prediction.size):
            if(prediction[i]==test_targets[i]): counter += 1
        class_rate = counter / test_targets.size
        class_list.append(class_rate)
    print(class_list)
    plt.plot(k_test,class_list,label="testing")
    plt.xlabel("k*")
    plt.ylabel("classification_rate")
    plt.xticks(k_test)
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
