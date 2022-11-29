# %%
import analysis_functions as funcs
import pandas as pd
import seaborn as sns

# %% 
# Read in the dataset
df = pd.read_csv("updated_datasets/merged_dataset.csv")

prediction_col = 'Life Expectancy'
id_cols = ['Entity', 'Year', 'Clusters'] # Identification columns
cancer_cols = [] # Columns related to cancer rates
causal_cols = [] # Columns related to causal factors for life expectancy
for col in df.columns:
    if "cancer" in col:
        cancer_cols.append(col)
    elif not (col in id_cols or col == prediction_col):
        causal_cols.append(col) 

# find # of clusters with silhoutette score. 1 -> best, -1 -> worst
df_single_year = funcs.filter_year(df, 2010)
funcs.plot_elbow_method_k_means(df_single_year, causal_cols, 12)

# %%
# Cancer to life expectancy relationship
life_expectancy_affecting_cancers = [
    "Pancreatic cancer",
    "Larynx cancer",
    "Brain and nervous system cancer"
]
regression_model = funcs.ols_regression_model(
    df, 
    life_expectancy_affecting_cancers, 
    prediction_col
)
print(regression_model.summary())

# %%
# Cancer to life expectancy relationship
# pancreatic causes
pancreatic_causes = [
    'Milk Consumption',
    'Mortality rate, under-5 (per 1,000 live births)',
    'Public Health Expenditure pc GDP'
]
regression_model = funcs.ols_regression_model(
    df, 
    pancreatic_causes, 
    "Pancreatic cancer"
)
print(regression_model.summary())

# %%
# Cancer to life expectancy relationship
# larynx causes
larynx_causes = [
    'Pigmeat Consumption',
    'Poultry Consumption',
    'Milk Consumption',
    'GDP per capita (2017 international $)'
]
regression_model = funcs.ols_regression_model(
    df, 
    larynx_causes, 
    "Larynx cancer"
)
print(regression_model.summary())

# %%
# Cancer to life expectancy relationship
# larynx causes
brain_causes = [
    'Pigmeat Consumption',
    'Poultry Consumption',
    'Obesity Rate',
    'GDP per capita (2017 international $)'
]
regression_model = funcs.ols_regression_model(
    df, 
    brain_causes, 
    "Brain and nervous system cancer"
)
print(regression_model.summary())


# %%
# get clusters based on best value of k
silhouette_scores = funcs.get_silhouette_score_df(df_single_year, causal_cols, 12)
best_k = silhouette_scores.sort_values(
    by="Silhouette Score for k(clusters)", 
    ascending=False
).iloc[0].name

df_single_year['Clusters'] = funcs.get_k_means_labels(df_single_year, causal_cols, best_k)
df_single_year['Clusters'].value_counts()

# %%
df_single_year[['Entity', 'Clusters'] + causal_cols].sort_values(by='Clusters')

# %%
sns.scatterplot(x="Public Health Expenditure pc GDP", y='Life Expectancy', hue='Clusters', data=df_single_year)


# %%
regression_model = funcs.ols_regression_model(df, causal_cols, "Life Expectancy")
print(regression_model.summary())

# %%
sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])


# %%
sns.scatterplot(x='Pigmeat Consumption', y='GDP per capita (2017 international $)', data=funcs.filter_year(df, 2000)[0])
# %%
sns.scatterplot(x='Beef and Buffallo Consumption', y='GDP per capita (2017 international $)', data=funcs.filter_year(df, 2000)[0])
# %%
