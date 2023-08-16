import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

def result_build(model: BaseEstimator,
                 X_test: pd.DataFrame,
                 Y_test: pd.Series,
                 df_index_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame containing model predictions, probabilities and true targets
    based on a predictive model and input data.

    Args:
        model (BaseEstimator): A trained predictive model
        X_test (pd.DataFrame): The input test dataset for making predictions.
        Y_test (pd.Series): The actual target values.
        df_index_cols (pd.DataFrame): Identifiers for each prediction, e.g. customer ID

    Returns:
        pd.DataFrame: A DataFrame with probabilities, predictions and true targets
            for each class
    """

    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)
    predictions2 = pd.Series(data=predictions, index=X_test.index, name='predicted_value')

    results = pd.DataFrame(data=probas, index=X_test.index, columns=model.classes_)
    results['predicted_value'] = predictions2
    results['target'] = Y_test
    
    results = pd.concat((df_index_cols, results), axis = 1, join = "inner")

    return results


def performance_metric(results_df: pd.DataFrame,
                       model: BaseEstimator) -> pd.DataFrame:
    """
    Calculate performance metrics for a predictive model based on predicted and
    true values.

    This function calculates various performance metrics, such as F-score, recall,
    and precision, for a given predictive model using predicted and true values
    from a DataFrame.

    Args:
        results_df (pd.DataFrame): A DataFrame containing predicted and true target
                                   values.
        model (BaseEstimator): A trained predictive model with a `classes_` attribute.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for each class
                      in the model.
    """
    model_scores = pd.DataFrame(model.classes_, columns=['product'])
    model_scores['f_score'] = f1_score(results_df["target"],
                                       results_df["predicted_value"],
                                       average=None)
    model_scores['recall'] = recall_score(results_df["target"],
                                          results_df["predicted_value"],
                                          average=None)
    model_scores['precision'] = precision_score(results_df["target"],
                                                results_df["predicted_value"],
                                                average=None)
    metrics = round(model_scores, 2)
    return metrics

def get_confusion_matrix(test_y: pd.DataFrame,
                         predictions: pd.DataFrame,
                         best_model: BaseEstimator) -> pd.DataFrame:
    """
    Calculate the confusion matrix for a set of predictions and true labels.

    This function calculates the confusion matrix for a given set of predicted
    labels and true labels.

    Args:
        test_y: The true labels.
        predictions: The predicted labels.
        best_model (BaseEstimator): The trained predictive model.

    Returns:
        pd.DataFrame: A DataFrame containing the confusion matrix.
    """
    
    cm = pd.DataFrame(confusion_matrix(test_y, predictions))
    cm.columns = best_model.classes_
    cm.columns.name = "Actual"
    cm.index = best_model.classes_
    cm.index.name = "Predicted"
    return cm

def get_distributions(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregated distributions based on the confusion matrix.

    This function calculates aggregated distributions for predicted and actual
    labels based on the provided confusion matrix.

    Args:
        cm (pd.DataFrame): The confusion matrix.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated distributions.
    """
    cm.columns.name = None
    cm.index.name = None
    preds = cm.sum(axis=0)
    preds_agg = (preds / sum(preds)).sort_values(ascending=False)
    
    actual = cm.sum(axis=1)
    actual_agg = (actual / sum(actual)).sort_values(ascending=False)

    preds_df = pd.DataFrame(preds_agg)
    preds_df.reset_index(level=0, inplace=True)

    act_df = pd.DataFrame(actual_agg)
    act_df.reset_index(level=0, inplace=True)

    distributions = pd.merge(preds_df, act_df, on='index')
    distributions.columns = ['class', 'model', 'actual']
    
    return distributions


def auc_plot(X_test, y_test, model):
    """
    Plot ROC curve and AUC scores for a predictive model.

    This function generates and plots the ROC curve along with AUC scores for a
    given predictive model using test data.

    Args:
        X_test: Test data features.
        y_test: Test data labels.
        model: The trained predictive model.

    Returns:
        None
    """
    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    # summarize scores
    NSAUC = 'AUC=%.3f' % ns_auc
    ROAUC = 'AUC=%.3f' % lr_auc
    
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label=('No Skill: ' + NSAUC))
    modname = str(model)
    plt.plot(lr_fpr, lr_tpr, marker='.', label=(modname + ': ' + ROAUC))
    
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # show the legend
    plt.legend(["Baseline", "Model"], loc="lower right")
    
    # show the plot
    plt.show()
