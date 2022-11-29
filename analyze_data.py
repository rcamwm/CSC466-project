# %%
import analysis_functions as funcs
import pandas as pd
import numpy as np
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

# %%
# Find which cancer types have biggest impact on life expectancy
# by calculating CV errors for different features
life_cancer_errors = pd.Series(
    dtype='float64'
)
# Cancer combinations limited to 3 due to computational limitations
cancer_combos = funcs.get_list_combinations(cancer_cols, 3)
for cancers in cancer_combos[1:]:
    life_cancer_errors[str(list(cancers))] = funcs.get_cv_average_error(
        df, 
        list(cancers),
        prediction_col
    )

# %%
# Find cancer types that most affect life expectancy
# The more affected it is by a set of cancers,
# the less the minimum error will be
min_cancer_errors = life_cancer_errors.min()

# %%
# Make new list of relevant cancers
# for all cancer types
relevant_cancers = life_cancer_errors.loc[
    life_cancer_errors == min_cancer_errors
].index[0].strip("[']").split("', '")

# %%
# Calculate average CV errors for different features
# for all different kinds of cancers
cancer_feature_errors = pd.DataFrame(
    columns=relevant_cancers, 
    dtype='float64'
)
# Causal combinations limited to 4 due to computational limitations
feature_combos = funcs.get_list_combinations(causal_cols, 4)
for features in feature_combos[1:]:
    cancer_feature_errors.loc[str(list(features))] = np.inf
    for cancer in relevant_cancers:
        cancer_feature_errors[cancer][str(list(features))] = funcs.get_cv_average_error(
            df, 
            list(features),
            cancer
        )

# %%
# Find cancer types that are most affected by selected features
# The more affected they are by a set of features,
# the less the minimum error will be
min_feature_errors = cancer_feature_errors.min().sort_values()
min_feature_errors

# %%
# Create new DataFrame to store features and k 
# for all cancer types
df_cancer_features_k = pd.DataFrame(
    index=min_feature_errors.index,
    columns=["Features", "k"]
)

# %%
# Get best feature set of all possible features
for col in cancer_feature_errors.columns:
    df_cancer_features_k["Features"][col] = cancer_feature_errors[col].loc[
        cancer_feature_errors[col] == cancer_feature_errors[col].min()
    ].index[0].strip("[']").split("', '")

# %%
# Find best value of k for k-nearest neighbors
df_cancer_features_k["k"] = df_cancer_features_k.apply(
    lambda cancer: funcs.get_best_k(
        df, 
        list(features), 
        cancer.name,
        50
    ), axis=1
)

# %%
# print strongest causes for each cancer type
for row in df_cancer_features_k.index:
    print("{}:".format(row))
    for feature in df_cancer_features_k["Features"][row]:
        print(feature)
    print()

# %%
# Create stacked models for each cancer
cancer_pred_models = {}
for cancer in df_cancer_features_k.index:
    cancer_pred_models[cancer] = funcs.get_stacking_model(
        df=df,
        X_cols=df_cancer_features_k["Features"][cancer], 
        y_col=cancer,
        neighbor_count=df_cancer_features_k["k"][cancer]
    )

# %%
# Get best value of k for k-nearest neighbors
# to predict life expectancy when using 
# most relevant cancer types
life_cancer_k = funcs.get_best_k(
    df, 
    relevant_cancers,  
    prediction_col,
    30
)

# %%
# Create stacked model for life expectancy based on cancers
life_cancer_model = funcs.get_stacking_model(
    df=df,
    X_cols=relevant_cancers, 
    y_col=prediction_col,
    neighbor_count=life_cancer_k
)

# %%

# %%
# TEST
def test(country, year):
    country_year_entry = df.loc[
        (df["Entity"] == country) &
        (df["Year"] == year)
    ]
    cancer_predictions = pd.DataFrame(
        columns=relevant_cancers,
        index=["Cancer Likelihood"]
    )
    for cancer in relevant_cancers:
        cancer_predictions[cancer]["Cancer Likelihood"] = (
            cancer_pred_models[cancer].predict(
                df[df_cancer_features_k["Features"][cancer]]
            ).mean()
        )
    life_exp_prediction = life_cancer_model.predict(
        cancer_predictions
    )[0]

    difference = life_exp_prediction - country_year_entry["Life Expectancy"].iloc[0]
    print("Expected life expectancy: {}".format(life_exp_prediction))
    print("Actual life expectancy: {}".format(country_year_entry["Life Expectancy"].iloc[0]))
    print("Off by: {}".format(difference))

# %%
test("Albania", 2010)

# %%
test("United States", 2017)

# # find # of clusters with silhoutette score. 1 -> best, -1 -> worst
# df_single_year = funcs.filter_year(df, 2010)
# funcs.plot_elbow_method_k_means(df_single_year, causal_cols, 12)

# %%
# get clusters based on best value of k
# silhouette_scores = funcs.get_silhouette_score_df(df_single_year, causal_cols, 12)
# best_k = silhouette_scores.sort_values(
#     by="Silhouette Score for k(clusters)", 
#     ascending=False
# ).iloc[0].name

# df_single_year['Clusters'] = funcs.get_k_means_labels(df_single_year, causal_cols, best_k)
# df_single_year['Clusters'].value_counts()

# # %%
# df_single_year[['Entity', 'Clusters'] + causal_cols].sort_values(by='Clusters')

# # %%
# sns.scatterplot(x="Public Health Expenditure pc GDP", y='Life Expectancy', hue='Clusters', data=df_single_year)


# # %%
# regression_model = funcs.ols_regression_model(df, causal_cols, "Life Expectancy")
# print(regression_model.summary())

# # %%
# sns.pairplot(df[['GDP per capita (2017 international $)', 'Life Expectancy']])


# # %%
