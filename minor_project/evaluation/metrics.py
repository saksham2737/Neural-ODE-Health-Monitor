from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    return acc, f1, auc