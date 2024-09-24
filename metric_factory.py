from scipy.stats import rankdata
import numpy as np
import sys

def fast_bin_auc(actual, predicted, partial=False):
    actual, predicted = actual.flatten(), predicted.flatten()
    if partial:
        n_nonzeros = np.count_nonzero(actual)
        n_zeros = len(actual) - n_nonzeros
        k = min(n_zeros, n_nonzeros)
        predicted = np.concatenate([
            np.sort(predicted[actual == 0])[::-1][:k],
            np.sort(predicted[actual == 1])[::-1][:k]
        ])
        actual = np.concatenate([np.zeros(k), np.ones(k)])

    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0: return 0
    return (np.sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def compute_sensitivity(actual, predicted):
    # Flatten the arrays to 1D
    actual = actual.flatten()
    predicted = predicted.flatten()

    # Calculate true positives and false negatives
    true_positives = np.sum((predicted == 1) & (actual == 1))
    false_negatives = np.sum((predicted == 0) & (actual == 1))

    # Compute sensitivity
    sensitivity = true_positives / (true_positives + false_negatives)
    
    return sensitivity


def fast_multiclass_dice(actual, predicted, n_classes):#comprovar que actuall i predicted nomes tenen bools
    #actual = np.squeeze(np.array(actual).astype(bool)) # bools are way faster to deal with by numpy
    #predicted = np.squeeze(np.array(predicted).astype(bool))
    actual = np.squeeze(np.array(actual))
    predicted = np.squeeze(np.array(predicted))
    print('actual shape:', actual.shape)
    print('actual dtype', actual.dtype)
    print('predicted shape:', predicted.shape)
    print('predicted dtype', predicted.dtype)

    # Initialize an array to store the dice score for each class
    dices = np.zeros(n_classes) 
    for cls in range(n_classes):
        actual_cls = (actual == cls)
        predicted_cls = (predicted == cls)
        actual_cls = np.array(actual_cls).astype(bool)
        predicted_cls = np.array(predicted_cls).astype(bool)
        print('actual_cls shape:', actual_cls.shape)
        print('actual_cls dtype', actual_cls.dtype)
        print('predicted_cls shape:', predicted_cls.shape)
        print('predicted_cls dtype', predicted_cls.dtype)
        
        intersections = np.logical_and(actual_cls, predicted_cls).sum(axis=(0, 1, 2))
        im_sums = actual_cls.sum(axis=(0, 1, 2)) + predicted_cls.sum(axis=(0, 1, 2))
        #intersections = np.logical_and(actual[0], predicted[0]).sum(axis=(0, 1, 2))
        #im_sums = actual[0].sum(axis=(0, 1, 2)) + predicted[0].sum(axis=(0, 1, 2))
        dices[cls] = 2. * intersections / np.maximum(im_sums, 1e-6)
    return dices

def binary_ECE(y_true, y_prob, power=1, bins=15):
    y_true, y_prob = y_true.flatten(), y_prob.flatten()

    idx = np.digitize(y_prob, np.linspace(0, 1, bins)) - 1

    def bin_func(p, y, idx):
        return (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(y_prob)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(y_prob, y_true, idx == i)
    return ece
