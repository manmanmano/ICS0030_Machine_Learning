#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd

# STUDENT SHALL ADD NEEDED IMPORTS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from common import describe_data, test_env
from common import classification_metrics as cm


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # Replace missing categorical values with common label
    for column in categorical_columns:
        df[column] = df[column].fillna(value='Missing')

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    for column in categorical_columns:
        df = pd.get_dummies(df, prefix=[column], columns=[column])

    # drop first column
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES
    df = df.fillna(value=0)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER
def logistic_regression(X, y):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = LogisticRegression(random_state=0, solver='sag')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(y_test, y_pred, label="Logistic Regression test data")


def k_nearest_neighbour(X, y):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(y_test, y_pred, label="KNN test data")


def svc(X, y):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = SVC(kernel='sigmoid', gamma=0.1, probability=True, tol=1e-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(y_test, y_pred, label="SVC test data")


def naive_bayes(X, y):
    sc = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(y_test, y_pred, label="Naive bayes test data")


def decision_tree_classifier(X, y):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(
        y_test, y_pred, label="Decision tree classifier test data")


def random_forest_classifier(X, y):
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = RandomForestClassifier(n_estimators=14)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm.print_metrics(
        y_test, y_pred, label="Random forest classifier test data")


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')
    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT
    describe_data.print_overview(
        students, file='results/students_overview.txt')
    describe_data.print_categorical(
        students, file='results/students_categorical_data.txt')

    # CREATE FILE WITH STUDENTS THAT WILL DROP OUT BY THE 4TH SEMESTER
    failing_students = students.loc[students['In university after 4 semesters'] == 'No']
    describe_data.print_overview(
        failing_students, file='results/failing_students_overview.txt')
    describe_data.print_categorical(
        failing_students, file='results/failing_students_categorical_data.txt')

    students_X, students_y = preprocess_data(students)

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    logistic_regression(students_X, students_y)
    k_nearest_neighbour(students_X, students_y)
    svc(students_X, students_y)
    naive_bayes(students_X, students_y)
    decision_tree_classifier(students_X, students_y)
    random_forest_classifier(students_X, students_y)

    print('Done')
