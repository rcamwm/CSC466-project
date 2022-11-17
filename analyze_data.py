# %%
import analysis_functions as funcs
import pandas as pd
import seaborn as sns

# %% 
# read in the dataset
df = pd.read_csv("updated_datasets/merged_dataset.csv")

'''
DataFrame columns - copy and paste as necessary

IDENTIFICATION:
'Entity',
'Year',
'Clusters',

STUDY DATA:
'Life Expectancy',
'Mutton and Goat Consumption', 
'Beef and Buffallo Consumption',
'Pigmeat Consumption', 
'Poultry Consumption', 
'Other Meats Consumption',
'Mortality rate, under-5 (per 1,000 live births)',
'GDP per capita (2017 international $)',
'Public Health Expenditure pc GDP',
'''

# %%
# DataFrame columns to study
study_cols = [
    'GDP per capita (2017 international $)', 
    'Public Health Expenditure pc GDP',
    'Beef and Buffallo Consumption'
]

# %%
# find # of clusters with silhoutette score. 1 -> best, -1 -> worst
df_single_year = funcs.filter_year(df, 2005)
funcs.plot_elbow_method_k_means(df_single_year, study_cols, 12)

# %%
# get clusters based on best value of k
silhouette_scores = funcs.get_silhouette_score_df(df_single_year, study_cols, 10)
best_k = silhouette_scores.sort_values(
    by="Silhouette Score for k(clusters)", 
    ascending=False
).iloc[0].name

df_single_year['Clusters'] = funcs.get_k_means_labels(df_single_year, study_cols, best_k)
df_single_year['Clusters'].value_counts()

# %%
sns.scatterplot(x="Pigmeat Consumption", y='Life Expectancy', hue='Clusters', data=df_single_year)


# %%
print(funcs.ols_regression_model_summary(df, study_cols, "Life Expectancy"))

# %%
sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])


# %%
