'''
This Script allows to produce a model for predicting if a customer
is fond of churning.

EDA plots are generated in order to understand relationships among variables.

Result plots include ROC and F1-Score.

Author: Alfonso Ponce
Date: 14/10/2023
'''

# import libraries

import os

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import constants as C
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            datafr: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(datafr):
    '''
    perform eda on datafr and save figures to images folder
    input:
            datafr: pandas dataframe

    output:
            None
    '''

    # EDA console prints
    print('===============================')
    print('Data first rows:')
    print(datafr.head())

    print('===============================')
    print(f'Data shape: {datafr.shape}')

    print('===============================')
    print('Number of null values per variable:')
    print(datafr.isnull().sum())

    print('===============================')
    print('Descriptive Statistics Summary')
    print(datafr.describe())

    # Creating and storing plots
    C.EDA_DATA_FOLDER.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(20, 10))
    datafr[C.RESPONSE_VARIABLE].hist()
    plt.savefig(str(C.EDA_DATA_FOLDER.joinpath(
        'Target_Variable_Histogram.png')))

    plt.figure(figsize=(20, 10))
    datafr['Customer_Age'].hist()
    plt.savefig(str(C.EDA_DATA_FOLDER.joinpath('Customer_Age_Histogram.png')))

    plt.figure(figsize=(20, 10))
    datafr.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(str(C.EDA_DATA_FOLDER.joinpath(
        'Marital_Status_Histogram.png')))

    plt.figure(figsize=(20, 10))
    sns.histplot(datafr['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(str(C.EDA_DATA_FOLDER.joinpath('Total_Trans_Ct_KDE.png')))

    plt.figure(figsize=(20, 10))
    sns.heatmap(datafr.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(str(C.EDA_DATA_FOLDER.joinpath('Heatmap.png')))


def encoder_helper(datafr, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            datafr: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name


    output:
            datafr: pandas dataframe with new columns for
    '''

    for column in category_lst:
        val_list = []
        gender_groups = datafr.groupby(column).mean()[response]

        for val in datafr[column]:
            val_list.append(gender_groups.loc[val])

        datafr[f'{column}_{response}'] = val_list

    return datafr


def perform_feature_engineering(datafr, response):
    '''
    input:
              datafr: pandas dataframe
              response: string of response name

    output:
              data_x_train: X training data
              data_x_test: X testing data
              data_y_train: y training data
              data_y_test: y testing data
    '''

    # dividing data into x and y
    datafr = encoder_helper(datafr, C.CAT_COLUMNS_LIST, response)

    data_y = datafr[C.RESPONSE_VARIABLE]
    data_x = pd.DataFrame()
    data_x[C.KEEP_COLS_LIST] = datafr[C.KEEP_COLS_LIST]

    # train test split
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=0.3, random_state=42)

    return data_x_train, data_x_test, data_y_train, data_y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    C.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(str(C.RESULTS_FOLDER.joinpath('Random_Forest_Results.png')))

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(str(C.RESULTS_FOLDER.joinpath(
        'Logistic_Regression_Results.png')))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def roc_comparison(model_1, model_2, x_test, y_test, out_path):
    '''
    Plots the ROC curves of the two models passed in the same figure.
    input:
            model_1: first model to evaluate
            model_2: second model to evaluate
            x_test: x testing data
            y_test: y testing data
            out_path: path to store the plot

    output:
            None

    '''
    plot_roc_curve(model_1, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        model_2,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8)
    plt.savefig(out_path)


def tree_explainer(tree_model, x_test, out_path):
    '''
    Plots SHAP values in a figure.

    input:
            tree_model: Tree-like model to perfor SHAP
            x_test: x testing data
            out_path: path to store the plot

    output:
            None
    '''
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.savefig(out_path)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # model training and predicting
    rfc = RandomForestClassifier(random_state=42)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # storing results in images
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    plt.clf()
    roc_comparison(lrc, cv_rfc.best_estimator_, x_test, y_test,
                   str(C.RESULTS_FOLDER.joinpath('ROC_Comparison.png')))

    plt.clf()
    tree_explainer(cv_rfc.best_estimator_, x_test, str(
        C.RESULTS_FOLDER.joinpath('Tree_Explainer.png')))

    plt.clf()
    feature_importance_plot(cv_rfc, x_train, str(
        C.RESULTS_FOLDER.joinpath('Feature_Importance.png')))

    # save best model
    C.MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
    joblib.dump(
        cv_rfc.best_estimator_,
        C.MODEL_FOLDER.joinpath('rfc_model.pkl'))
    joblib.dump(lrc, C.MODEL_FOLDER.joinpath('logistic_model.pkl'))


if __name__ == '__main__':
    dataframe = import_data(str(C.DATA_PATH))
    dataframe[C.RESPONSE_VARIABLE] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(dataframe)
    x_train_data, x_test_data, y_train_data, y_test_data = perform_feature_engineering(
        dataframe, C.RESPONSE_VARIABLE)
    train_models(x_train_data, x_test_data, y_train_data, y_test_data)
