import numpy as np

def accuracy(y_pred, y_actual):
    """
    Calculates the accuracy of the model.

    Args:
        y_pred (ndarray): Predicted class labels.
        y_actual (ndarray): Actual class labels.

    Returns:
        float: Accuracy of the model.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    return (y_pred == y_actual).sum()/len(y_pred)

def precision(y_pred, y_actual):
    """
    Calculates the precision of the model.

    Args:
        y_pred (ndarray): Predicted class labels.
        y_actual (ndarray): Actual class labels.

    Returns:
        float: Precision of the model.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    tp = ((y_pred == 1) & (y_actual == 1)).sum()
    fp = ((y_pred == 1) & (y_actual == 0)).sum()
    return tp/(tp+fp)

def recall(y_pred, y_actual):
    """
    Calculates the recall of the model.

    Args:
        y_pred (ndarray): Predicted class labels.
        y_actual (ndarray): Actual class labels.

    Returns:
        float: Recall of the model.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    tp = ((y_pred == 1) & (y_actual == 1)).sum()
    fn = ((y_pred == 0) & (y_actual == 1)).sum()
    return tp/(tp+fn)

def f1_score(y_pred, y_actual):
    """
    Calculates the f1 score of the model.

    Args:
        y_pred (ndarray): Predicted class labels.
        y_actual (ndarray): Actual class labels.

    Returns:
        float: F1 score of the model.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    p = precision(y_pred, y_actual)
    r = recall(y_pred, y_actual)
    return 2*p*r/(p+r)

def confusion_matrix(y_pred, y_actual):
    """
    Calculates the confusion matrix of the model.

    Args:
        y_pred (ndarray): Predicted class labels.
        y_actual (ndarray): Actual class labels.

    Returns:
        ndarray: Confusion matrix of the model.
    """
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    tp = ((y_pred == 1) & (y_actual == 1)).sum()
    fp = ((y_pred == 1) & (y_actual == 0)).sum()
    fn = ((y_pred == 0) & (y_actual == 1)).sum()
    tn = ((y_pred == 0) & (y_actual == 0)).sum()
    return np.array([[tp, fp], [fn, tn]])