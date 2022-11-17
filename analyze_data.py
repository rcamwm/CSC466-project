# %%
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.cluster as cluster

import data_functions as funcs

# %% 
# read in the dataset
df = pd.read_csv("updated_datasets/merged_dataset.csv")

# %%
meat_consumption = [
    "Mutton and Goat Consumption", 
    "Beef and Buffallo Consumption",
    "Pigmeat Consumption", 
    "Poultry Consumption", 
    "Other Meats Consumption"
]
print(funcs.ols_regression_model_summary(df, meat_consumption, "Life Expectancy"))


# %%
sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])

# %%
# find # of clusters with silhoutette score. 1 -> best, -1 -> worst
df_short = df.loc[
    df['Year'] == 2005
][['GDP per capita (2017 international $)', 'Life Expectancy']]
funcs.plot_elbow_method_k_means(df_short, 12)

# %%
# print out the silhotette score for cluster 3 - 20
funcs.get_silhouette_score_df(df_short, 20)

# %%

# plot the clusters based on the best result from the previous step.
df['Clusters'] = funcs.get_k_means_labels(df, ['Life Expectancy', 'GDP per capita (2017 international $)'], 3)
df['Clusters'].value_counts()

# %%
