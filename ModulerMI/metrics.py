import numpy as np
from sklearn.metrics import accuracy_score

def classification_metric(predicted, true, bins):
    if predicted.shape[0] != true.shape[0]:
        raise ValueError("Predicted and true arrays must have the same length.")

    predicted_classes = np.digitize(predicted, bins) - 1
    true_classes = np.digitize(true, bins) - 1

    num_bins = len(bins) - 1
    predicted_classes[predicted_classes == num_bins] = num_bins - 1
    true_classes[true_classes == num_bins] = num_bins - 1

    accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy

def multivariate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    mean = np.mean(y_true)
    ss_tot = np.sum((y_true - mean) ** 2)
    return 1 - (ss_res / ss_tot)
