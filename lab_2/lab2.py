#!/usr/bin/env python3
"""lab2 template"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from common import feature_selection as feat_sel
from common import test_env


# Pylint error W0621: Redefining name 'X' from outer scope (line 61) (redefined-outer-name)
# pylint: disable=redefined-outer-name


def print_metrics(y_true, y_pred, label):
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R squared: %.2f' % (label, r2_score(y_true, y_pred)))


# linear regression 5
def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Linear regression all in
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(
        X_sel, y, print_text='Linear regression with feature selection')


# Polynomial regression
def polynomial_regression_selection(X, y, print_text='Polynomial regression'):
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = linear_regression(X_poly, y, print_text)
    lin_reg.fit(X, y)
    return poly_reg


def svr_regression_selection(X, y, print_text='SVR'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = sc.fit_transform(np.expand_dims(y, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = SVR(kernel='rbf', gamma='auto')
    reg.fit(X_train, y_train)
    print_metrics(np.squeeze(y_test), np.squeeze(
        reg.predict(X_test)), print_text)
    return reg


def decision_tree_regression_selection(X, y, print_text='Decision tree regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = DecisionTreeRegressor(max_depth=None, min_samples_split=2,
                                min_samples_leaf=1)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def random_forest_regression_selection(X, y, print_text='Random forest regression'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    reg = RandomForestRegressor(n_estimators=10)
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


if __name__ == '__main__':
    test_env.versions(['numpy', 'statsmodels', 'sklearn'])

    # https://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset
    # matrix of features X and dependent variable Y
    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    polynomial_regression_selection(X, y)
    svr_regression_selection(X, y)
    decision_tree_regression_selection(X, y)
    random_forest_regression_selection(X, y)

    print('Done')
