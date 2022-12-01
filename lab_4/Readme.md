# Dataset 

The dataset was found on Kaggle. It has the purpose to categorize the countries using socio-economic and health factors the determine how developed a country is.
Dataset can be found by clicking on the link below:
https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?select=Country-data.csv

# Hypothesis

The datasets has many interesting columns, but two in particular caught my attention: 
    - health: country expenses on health (%age of GDP)
    - life_expec: avg number of years a person could live with current mortality patterns (1-100, float)

My hypothesis is the following:
##### The higher are the expenses on health system in a country, the longer is its people's life expectation

# Algorithm and Used Libraries

In order to get an answer I used the k-means algorithm.
Libraries used in the project are: sklearn and matplotlib.

# Result
K-means does the job well, separating the 4 clusters (established with the elbow method) cleanly and precisely.
From the file (results/kmeans.png) we can see clearly that in most of the cases a more advanced health system can make people's lives longer, but it is not always the case as we see for the first cluster (the one on the left). 
One possibility can be, that even if the countries do not have high expenses on health, the quality of life in the clustered countries can be high enough to expand people's life. There is also the possibility that in poorer and more "dangerous" countries, people have adapted and become stronger.
Finally, we can say the hypothesis is partially correct. 
It would have been fully correct if the first cluster centroid would have been right before the highest one.
