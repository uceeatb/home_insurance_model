import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xgboost as xgb
import shap

class UpliftChart():

    def cumulative_gain_curve(y_true, y_score, pos_label=None):

        """This function generates the points necessary to plot the Cumulative Gain
        Note: This implementation is restricted to the binary classification task.
        Args:
            y_true (array-like, shape (n_samples)): True labels of the data.
            y_score (array-like, shape (n_samples)): Target scores, can either be
                probability estimates of the positive class, confidence values, or
                non-thresholded measure of decisions (as returned by
                decision_function on some classifiers).
            pos_label (int or str, default=None): Label considered as positive and
                others are considered negative
        Returns:
            percentages (numpy.ndarray): An array containing the X-axis values for
                plotting the Cumulative Gains chart.
            gains (numpy.ndarray): An array containing the Y-axis values for one
                curve of the Cumulative Gains chart.
        Raises:
            ValueError: If `y_true` is not composed of 2 classes. The Cumulative
                Gain Chart is only relevant in binary classification.
        """
        y_true, y_score = np.asarray(y_true), np.asarray(y_score)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        sorted_indices = np.argsort(y_score)[::-1]
        y_true = y_true[sorted_indices]
        gains = np.cumsum(y_true)

        percentages = np.arange(start=1, stop=len(y_true) + 1)

        gains = gains / float(np.sum(y_true))
        percentages = percentages / float(len(y_true))

        gains = np.insert(gains, 0, [0])
        percentages = np.insert(percentages, 0, [0])

        return percentages, gains


    def plot_cumulative_gain(y_true_list, y_probas_list, label_list,
                             title=None,
                             ax=None, figsize=None, title_fontsize=20,
                             text_fontsize=12):

        """Generates the Cumulative Gains Plot from labels and scores/probabilities
        The cumulative gains chart is used to determine the effectiveness of a
        binary classifier. A detailed explanation can be found at
        http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
        here works only for binary classification.
        Args:
            y_true (array-like, shape (n_samples)):
                Ground truth (correct) target values.
            y_probas (array-like, shape (n_samples, n_classes)):
                Prediction probabilities for each class returned by a classifier.
            title (string, optional): Title of the generated plot. Defaults to
                "Cumulative Gains Curve".
            ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
                plot the learning curve. If None, the plot is drawn on a new set of
                axes.
            figsize (2-tuple, optional): Tuple denoting figure size of the plot
                e.g. (6, 6). Defaults to ``None``.
            title_fontsize (string or int, optional): Matplotlib-style fontsizes.
                Use e.g. "small", "medium", "large" or integer-values. Defaults to
                "large".
            text_fontsize (string or int, optional): Matplotlib-style fontsizes.
                Use e.g. "small", "medium", "large" or integer-values. Defaults to
                "medium".
        Returns:
            ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
                drawn.
        Example:
            >>> import scikitplot as skplt
            >>> lr = LogisticRegression()
            >>> lr = lr.fit(X_train, y_train)
            >>> y_probas = lr.predict_proba(X_test)
            >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
            >>> plt.show()
            .. image:: _static/examples/plot_cumulative_gain.png
               :align: center
               :alt: Cumulative Gains Plot
        """

        for i, label in enumerate(label_list):

            y_true = np.array(y_true_list[i])
            y_probas = np.array(y_probas_list[i])

            classes = np.unique(y_true)

            if len(classes) != 2:
                raise ValueError('Cannot calculate Cumulative Gains for data with '
                                 '{} category/ies'.format(len(classes)))

            # Compute Cumulative Gain Curves
            percentages, gains = UpliftChart.cumulative_gain_curve(y_true,
                                                                   y_probas[:, 1],
                                                                   classes[1])

            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)

            ax.set_title(title, fontsize=title_fontsize)

            percentages = percentages * 100
            gains = gains*100

            ax.plot(percentages, gains, lw=3, label='{}'.format(label))

        ax.plot([0, 100], [0, 100], 'k--', lw=2, label='Baseline')


        ax.set_xlabel('Percentage of population', fontsize=text_fontsize)
        ax.set_ylabel('Percentage Gain', fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
        ax.grid('on')
        ax.legend(loc='lower right', fontsize=15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        plt.show()
        
        return

class ProbaDist():
    
    def plot_predictions_distribution(predictions: np.array) -> None:
        """_summary_

        Args:
            df (DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        classes = list(predictions.columns)
        fig, ax = plt.subplots(len(classes), 1, figsize=(8, 4 * len(classes)))

        for i, class_name in enumerate(classes):

            if len(classes) == 1:
                temp_ax = ax
            else:
                temp_ax = ax[i]
            
            temp_ax.hist(
                predictions[class_name],
                bins = 50,
                histtype='bar',
                density = True
            )
            
            temp_ax.set_xlabel("Probability")
            temp_ax.set_ylabel("Density")
            temp_ax.legend(["class: "+str(class_name)])
        fig.tight_layout()
        plt.show()

        return

class FeatureImportance():
    
    def plot_feature_importance(model: xgb.sklearn.XGBClassifier,
                                features_list: list):
        """_summary_

        Args:
            df (DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        model.get_booster().feature_names = features_list

        fig, ax = plt.subplots(1,1,figsize=(10,20))
        plot = xgb.plot_importance(model,
                                   ax=ax)
        ax.grid(False)
        plt.show()

        return
    
    def plot_SHAP_summary(model: xgb.sklearn.XGBClassifier,
                          df: pd.DataFrame,
                          features_list: list,
                          max_display: int=20):
        """_summary_

        Args:
            df (DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        summary_explainer = shap.TreeExplainer(model)
        summary_shap_values = summary_explainer.shap_values(df[features_list])

        shap_plot = shap.summary_plot(summary_shap_values,
                                      df[features_list],
                                      max_display=max_display)

        return shap_plot
    
    def get_SHAP_explainer_values(model: xgb.sklearn.XGBClassifier,
                                  df: pd.DataFrame,
                                  features_list: list) -> np.array:
        """_summary_

        Args:
            df (DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        if len(df) > 10000:  
            df = df.sample(10_000)
        full_explainer = shap.TreeExplainer(model, df[features_list])
        full_shap_values = full_explainer(df[features_list])

        return full_shap_values

    def plot_SHAP_waterfall(shap_values: pd.DataFrame,
                            individual_plots: int=1):
        """_summary_

        Args:
            df (DataFrame): _description_

        Returns:
            DataFrame: _description_
        """
        if individual_plots > 10:
            individual_plots = 10

        shap_values = shap_values[0:individual_plots]

        fig, ax = plt.subplots(individual_plots, 1, figsize=(8, 4 * individual_plots))

        for i in range(individual_plots):
            shap.waterfall_plot(shap_values[i])
        fig.tight_layout()
        plt.show()

        return