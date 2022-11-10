# %%
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.cluster as cluster

# %%
def show_milk_breastcancer_graph(df, year):
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Milk Consumption",
        y="Breast Cancer Rates",
        title="Breast Cancer Rates in {}".format(year)
    )
    
# %%
def show_soy_breastcancer_graph(df, year):
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Soy Consumption",
        y="Breast Cancer Rates",
        title="Breast Cancer Rates in {}".format(year)
    )

# %%
def show_mutton_and_goat_graph(df, year):
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Mutton and Goat Consumption",
        y="Life Expectancy",
        title="Life Expectancy in {}".format(year)
    )

# %% 
df = pd.read_csv("updated_datasets/merged_dataset.csv")

# %%
x = df[["Mutton and Goat Consumption", "Beef and Buffallo Consumption", "Pigmeat Consumption", "Poultry Consumption", "Other Meats Consumption"]]
y = df["Life Expectancy"]
regr = linear_model.LinearRegression()
regr.fit(x, y)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print_model = model.summary()
print(print_model)

# %%
class object1:
    def __init__(self, entity: str, year: int, gpa_per_capita: float) -> None:
        self.entity = entity
        self.year = year
        self.gpa_per_capita = gpa_per_capita
object1_list = []
df.apply(lambda row : object1_list.append(object1(row['Entity'], row['Year'], row['GDP per capita (2017 international $)'])), axis = 1)

# %%
sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])

# find # of clusters
# %%
K = range(1, 12)
wss = []
df_short = df[['Life Expectancy', 'GDP per capita (2017 international $)']]
for k in K:
    kmeans = cluster.KMeans(n_clusters=k, init='k-means++')
    kmeans = kmeans.fit(df_short)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
sns.lineplot(x='Clusters', y='WSS', data=mycenters, marker='*')

# %%
for i in range(3, 13):
    labels=cluster.KMeans(n_clusters=i, init='k-means++', random_state=200).fit(df_short).labels_
    print("Silhouette score for k(clusters) = " + str(i) + " is "
        + str(metrics.silhouette_score(df_short, labels, metric='euclidean', sample_size=1000, random_state=200)))

# k-means clustering
# %%
kmeans = cluster.KMeans(n_clusters=3, init='k-means++')
kmeans = kmeans.fit(df[['Life Expectancy', 'GDP per capita (2017 international $)']])
kmeans.cluster_centers_
df['Clusters'] = kmeans.labels_
df.head()
df['Clusters'].value_counts()
sns.scatterplot(x="GDP per capita (2017 international $)", y='Life Expectancy', hue='Clusters', data=df)

# %%
