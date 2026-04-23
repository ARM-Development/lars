import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

def plot_confusion_matrix(df, label_col='label', pred_col='llm_label', normalize=None, ax=None):
    """
    Plot a confusion matrix using true and predicted labels from a DataFrame.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing true and predicted labels.
    label_col (str): Column name for true labels.
    pred_col (str): Column name for predicted labels.
    normalize (str or None): Normalization mode for confusion matrix.
    ax (matplotlib axis handle): The axis handle to plot on. Set to None to use the current axis.

    Returns
    -------
    None
    """
    le = LabelEncoder()
    true_labels = le.fit_transform(df[label_col].str.lower())
    pred_labels = le.transform(df[pred_col].str.lower())
    if ax is None:
        ax = plt.gca()

    cm = confusion_matrix(true_labels, pred_labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')


def calculate_cohen_kappa(df, label_col='label', pred_col='llm_label'):
    """
    Calculate Cohen's kappa from true and predicted labels in a DataFrame.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing true and predicted labels.
    label_col (str): Column name for true labels.
    pred_col (str): Column name for predicted labels.

    Returns
    -------
    float: Cohen's kappa coefficient.
    """
    return cohen_kappa_score(
        df[label_col].str.lower(),
        df[pred_col].str.lower(),
    )
