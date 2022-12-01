#!/usr/bin/env python3
"""lab1 template"""
import sys

import pandas as pd
from matplotlib import pyplot as plt

from common import describe_data
from common import test_env


def read_data(file):
    """Return pandas dataFrame read from csv file"""
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


if __name__ == '__main__':
    REQUIRED = ['matplotlib', 'pandas']
    test_env.versions(REQUIRED)
    df = read_data('data/breach_report.csv')

    # Print data overview with function print_overview in
    # common/describe_data.py
    describe_data.print_overview(df)

    # Remove maximum printed rows limit. Otherwise, next print can be truncated
    pd.options.display.max_rows = None

    # Print all possible values with counts in column 'State' with help of
    # groupby() and size()
    print(df.groupby('State').size(), '\n')

    # Print all unique values in column 'Type'
    print('Type of Breach:', df['Type of Breach'].unique(), '\n')

    # Replace all nan values with Unknown
    # NB! Missing data handling will be covered in 3rd class later on in course
    df = df.fillna(value='Unknown')

    # Extract Texas state (TX) data where incident type includes Hacking/IT Incident
    texas_hacking = df[(df['State'] == 'TX') & (
        df['Type of Breach'].str.contains('Hacking/IT Incident'))]

    # Extract column Individuals Affected to variable individuals_affected
    individuals_affected = texas_hacking['Individuals Affected']

    # Calculate affected individuals per breach mean to variable individuals_affected_mean and
    # print the value
    individuals_affected_mean = individuals_affected.mean()
    print('Texas hacking affected individuals per breach mean:',
          individuals_affected_mean)

    # Calculate median to variable individuals_affected_median and print the value
    individuals_affected_median = individuals_affected.median()
    print('Texas hacking affected individuals per breach median:',
          individuals_affected_median)

    # Calculate mode to variable individuals_affected_mode and print the value
    individuals_affected_mode = individuals_affected.mode()
    print('Texas hacking affected individuals per breach mode:',
          individuals_affected_mode)

    # Calculate standard deviation to variable individuals_affected_std and print the
    # value
    individuals_affected_std = individuals_affected.std()
    print('Texas hacking affected individuals per breach standard deviation:',
          individuals_affected_mode)

    # Calculate and print quartiles
    individuals_affected_quartiles = texas_hacking['Individuals Affected'].quantile([
                                                                                    0.25, 0.5, 0.75])
    print('Texas hacking affected individuals quartiles:',
          individuals_affected_quartiles)

    # Draw dataset histogram including mean, median and mode
    figure_1 = 'Texas hacking affected individuals per breach histogram'
    plt.figure(figure_1)
    plt.hist(
        individuals_affected,
        range=(
            individuals_affected.min(),
            individuals_affected.max()),
        bins=20,
        edgecolor='black')
    plt.title(figure_1)
    plt.xlabel('Affected individuals')
    plt.ylabel('Count')
    plt.axvline(individuals_affected_mean, color='r', linestyle='solid',
                linewidth=1, label='Mean')
    plt.axvline(individuals_affected_median, color='y', linestyle='dotted',
                linewidth=1, label='Median')
    plt.axvline(individuals_affected_mode[0], color='orange',
                linestyle='dashed', linewidth=1, label='Mode')

    plt.legend()
    plt.savefig('results/figure_1.png')

    # Draw boxplot
    figure_2 = 'Texas hacking affected individuals per breach box plot'
    plt.figure(figure_2)
    plt.boxplot(individuals_affected)
    plt.ylabel('Affected individuals')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.title(figure_2)
    plt.savefig('results/figure_2.png')

    # Show all figures in different windows
    plt.show()
    print('Done')
