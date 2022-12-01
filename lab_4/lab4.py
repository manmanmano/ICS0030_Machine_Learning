#!/usr/bin/env python3

# BUSINESS PROBLEM: Find correlation between the percentage of money spent on health by each country and the
#                   respective country's life expectancy

import sys

import pandas as pd

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from common import describe_data, test_env


def read_data(file):
    """Return pandas dataFrame read from csv file"""
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df):
    drop_columns = [
        'country', 'child_mort', 'exports', 'imports',
        'income', 'inflation', 'total_fer', 'gdpp']

    return df.drop(drop_columns, axis=1)


def plot_kmeans(X, y, n_clusters, title, x_label, y_label, k_means=None):
    colors = ['orange', 'green', 'brown', 'purple']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    for cluster in range(0, n_clusters):
        plt.scatter(X[y == cluster, 0], X[y == cluster, 1],
                    s=40, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    if k_means is not None:
        # Cluster centres only exist if clusterer is k_means
        plt.scatter(k_means.cluster_centers_[:, 0],
                    k_means.cluster_centers_[:, 1],
                    s=20, c='blue', label='Centroid')
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig('results/kmeans.png')
    plt.clf()


def plot_dataset_tsne(X, figure, file=''):
    plt.scatter(X[:, 0], X[:, 1], s=20)
    plt.title(figure)
    plt.xticks([])
    plt.yticks([])
    if file:
        plt.savefig(file)
    # plt.show()
    plt.savefig(file)
    plt.clf()


def plot_kmeans_tsne(X, y, figure, file=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    plt.figure(figure)

    for cluster in range(0, len(set(y))):
        plt.scatter(X[y == cluster, 0], X[y == cluster, 1],
                    s=20, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    plt.title(figure)
    # Remove axes numbers because those are not relevant for visualisation
    plt.xticks([])
    plt.yticks([])
    if file:
        plt.savefig(file)
    # plt.show()
    plt.savefig(file)
    plt.clf()


def cluster_data(df):

    # create 2d array of dataframe
    X = df.iloc[:, [1, 0]].values


    # use the elbow method to find the optimal number of clusters
    wcss = []
    max_clusters = 20
    for i in range(1, max_clusters):
        k_means = KMeans(n_clusters=i, init='k-means++', random_state=0)
        k_means.fit(X)
        wcss.append(k_means.inertia_)

    # save plotted elbow to results
    plt.plot(range(1, max_clusters), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('results/elbow.png')
    plt.clf()

    # visualize dataset with help of tsne dimensions reduction to 2 dimensions
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    plot_dataset_tsne(X_tsne, 'Dataset clusters with T-SNE',
                      'results/dataset_tsne.png')

    # plot clustered data
    n_clusters = 4
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    y_kmeans = k_means.fit_predict(X)

    # function in common/describe_clusters.py
    plot_kmeans(X, y_kmeans,
                n_clusters,
                'Countries life expectancy based on health expenses',
                'Life expectancy (1-100)',
                'Expenses on health (%)',
                k_means=k_means)

    # visualize kmeans with help of t-SNE dimensions reduction to 2 dimensions
    # function in common/describe_clusters.py
    plot_kmeans_tsne(X_tsne, y_kmeans,
                     'K-MEANS clusters with T-SNE', 'results/kmeans_tsne.png')


if __name__ == '__main__':
    # print out python and modules versions
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    # read dataset file to pandas data frame
    countries = read_data('data/country-data.xls')

    # save dataset description to file in results directory
    describe_data.print_overview(
        countries, file='results/countries_overview.txt')
    describe_data.print_categorical(
        countries, file='results/countries_categorical_data.txt')

    # drop every column besides health and life_expec
    preprocessed_countries = preprocess_data(countries)

    cluster_data(preprocessed_countries)

    print('Done')
