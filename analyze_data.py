# %%
import analysis_functions as funcs
import pandas as pd
import seaborn as sns

# %% 
# Read in the dataset
df = pd.read_csv("updated_datasets/merged_dataset.csv")

prediction_col = 'Life Expectancy'
study_cols = [
    'Mutton and Goat Consumption', 
    'Beef and Buffallo Consumption',
    'Pigmeat Consumption', 
    'Poultry Consumption', 
    'Other Meats Consumption',
    'Mortality rate, under-5 (per 1,000 live births)',
    'GDP per capita (2017 international $)',
    'Public Health Expenditure pc GDP'
]

'''
IDENTIFICATION COLUMNS:
'Entity',
'Year',
'Clusters',
'''

df_2017, df_training = funcs.filter_year(df, 2017)

# %%
# Calculate average CV errors for different features
feature_errors = pd.Series(dtype='float64')
for features in funcs.get_list_combinations(study_cols)[1:]:
    feature_errors[str(list(features))] = funcs.get_cv_average_error(
        df, 
        list(features),
        prediction_col
    )

# %%
# Get best feature set of all possible features
best_features = list(feature_errors.loc[
    feature_errors == feature_errors.min()
].index)[0].strip("[']").split("', '")
best_features

# %%
# Calculate average CV errors for different values of k
k_errors = pd.Series(dtype='float64')
for k in range(1, 50):
    k_errors[str(k)] = funcs.get_cv_error_for_k(
        df, 
        list(features),
        prediction_col,
        k
    )
k_errors.plot.line()

# %%
# Find best value of k for k-nearest neighbors
best_k = int(k_errors.loc[
    k_errors == k_errors.min()
].index.values)
best_k

# %%
df_2017, df_training = funcs.filter_year(df, 2017)
model = funcs.get_stacking_model(
    df=df,
    X_cols=best_features, 
    y_col=prediction_col,
    neighbor_count=best_k
)

# # %%
# # find # of clusters with silhoutette score. 1 -> best, -1 -> worst
# df_single_year = funcs.filter_year(df, 2010)
# funcs.plot_elbow_method_k_means(df_single_year, study_cols, 12)

# %%
# get clusters based on best value of k
# silhouette_scores = funcs.get_silhouette_score_df(df_single_year, study_cols, 12)
# best_k = silhouette_scores.sort_values(
#     by="Silhouette Score for k(clusters)", 
#     ascending=False
# ).iloc[0].name

# df_single_year['Clusters'] = funcs.get_k_means_labels(df_single_year, study_cols, best_k)
# df_single_year['Clusters'].value_counts()

# # %%
# df_single_year[['Entity', 'Clusters'] + study_cols].sort_values(by='Clusters')

# # %%
# sns.scatterplot(x="Public Health Expenditure pc GDP", y='Life Expectancy', hue='Clusters', data=df_single_year)


# # %%
# regression_model = funcs.ols_regression_model(df, study_cols, "Life Expectancy")
# print(regression_model.summary())

# # %%
# sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])


# # %%
