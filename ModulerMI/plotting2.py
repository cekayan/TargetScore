import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_multidimensional_roc(y_pred, y_actual, intervals):
    y_pred_classes = np.digitize(y_pred, intervals) - 1
    y_actual_classes = np.digitize(y_actual, intervals) - 1
    num_bins = len(intervals) - 1
    y_pred_classes[y_pred_classes == num_bins] = num_bins - 1
    y_actual_classes[y_actual_classes == num_bins] = num_bins - 1

    plt.figure(figsize=(10, 7))
    labels = [f"Class {i}" for i in range(num_bins)]
    for class_id, lab in enumerate(labels):
        y_actual_binary = (y_actual_classes == class_id).astype(int).ravel()
        y_pred_binary = (y_pred_classes == class_id).astype(int).ravel()
        fpr, tpr, _ = roc_curve(y_actual_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{lab}, AUC: {np.round(roc_auc, 2)}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    plt.title('ROC Curve for Different Intervals')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def classify_predicted_true(predicted, true, bins, labels=None, plot_title="Confusion Matrix"):
    predicted = np.array(predicted)
    true = np.array(true)
    if predicted.shape[0] != true.shape[0]:
        raise ValueError("Predicted and true arrays must have the same length.")

    predicted_classes = np.digitize(predicted, bins) - 1
    true_classes = np.digitize(true, bins) - 1
    num_bins = len(bins) - 1
    predicted_classes[predicted_classes == num_bins] = num_bins - 1
    true_classes[true_classes == num_bins] = num_bins - 1

    if labels is None:
        labels = [f"Class {i}" for i in range(num_bins)]
    elif len(labels) != num_bins:
        raise ValueError("Number of labels must match number of bins minus one.")

    accuracy = accuracy_score(true_classes, predicted_classes)
    cm = confusion_matrix(true_classes, predicted_classes)
    row_acc = np.diag(cm) / np.sum(cm, axis=1)
    col_acc = np.diag(cm) / np.sum(cm, axis=0)
    cm_ext = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1), dtype=float)
    cm_ext[:-1, :-1] = cm
    cm_ext[:-1, -1] = row_acc * 100
    cm_ext[-1, :-1] = col_acc * 100
    cm_ext[-1, -1] = accuracy * 100

    mask = np.zeros_like(cm_ext, dtype=bool)
    mask[-1, :-1] = True
    mask[:-1, -1] = True
    mask[-1, -1] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ext, annot=True, fmt='.1f', cmap='Reds', mask=~mask, cbar=False,
                xticklabels=labels + ['Row Acc (%)'], yticklabels=labels + ['Col Acc (%)'])
    sns.heatmap(cm_ext, annot=True, fmt='.1f', cmap='Blues', mask=mask, cbar=False,
                xticklabels=labels + ['Row Acc (%)'], yticklabels=labels + ['Col Acc (%)'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f"{plot_title}\nOverall Accuracy: {accuracy * 100:.2f}%")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.gca().xaxis.tick_top()
    plt.show()
    return accuracy
