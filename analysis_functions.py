import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import statsmodels.api as sm

from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_cv_average_error(df, features, prediction):
    '''
    Find the average error when using cross validation
    on a DataFrame, given a list of features and a 
    column to predict

    df : pandas.DataFrame
    features : list[], each entry should match to a column in df
    prediction : str, should match to a column in df
    '''
    pipeline = make_pipeline(
        StandardScaler(),
        # 4 nearest neighbors is arbitrary starting point
        KNeighborsRegressor(n_neighbors=4)
    )
    cv_errs = -cross_val_score(
        pipeline, 
        X=df[features],
        y=df[prediction],
        scoring="neg_mean_squared_error", 
        cv=10
    )
    return cv_errs.mean()

def get_cv_error_for_k(df, features, prediction, k):
    '''
    Calculate average error test error for a value of k
    when using cross validation on a DataFrame, 
    given features and a column to predict

    df : pandas.DataFrame
    features : list[], each entry should match to a column in df
    prediction : str, should match to a column in df
    k : int, a value of k for k-nearest neighbors
    '''
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k)
    )

    # calculate errors from cross-validation
    cv_errs = -cross_val_score(
        pipeline, 
        X=df[features], 
        y=df[prediction],
        scoring="neg_mean_squared_error", 
        cv=10
    )
    # calculate average of the cross-validation errors
    return cv_errs.mean()

def get_best_k(df, features, prediction, max_k):
    '''
    Calculate average CV errors for different values of k

    df : pandas.DataFrame
    features : list[], each entry should match to a column in df
    prediction : str, should match to a column in df
    max_k : int, the maximum possible value of k for k-nearest neighbors
    '''
    k_errors = pd.Series(dtype='float64')
    for k in range(2, max_k): 
        k_errors[str(k)] = get_cv_error_for_k(
            df, 
            features,
            prediction,
            k
        )
    return int(k_errors.loc[
        k_errors == k_errors.max()
    ].index.values[0])

def get_stacking_model(df, X_cols, y_col, neighbor_count):
    '''
    Creates a stacked model
    '''
    linear_model = LinearRegression()
    linear_model.fit(X=df[X_cols], y=df[y_col])

    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=neighbor_count)
    )
    knn_model.fit(X=df[X_cols], y=df[y_col])

    stacking_model = StackingRegressor([
        ("linear", linear_model), 
        ("knn", knn_model)],
        final_estimator=LinearRegression()
    )
    stacking_model.fit(X=df[X_cols], y=df[y_col])
    return stacking_model

def filter_year(df, year):
    '''
    Returns two pandas DataFrames.
    First DataFrame has data corresponding to a single year,
    Second DataFrame has data corresponding to all data EXCEPT that year

    df : pandas.DataFrame
    year : int, year to return data for, must be at least 1990 and at most 2017
    '''
    return df.loc[df['Year'] == year], df.loc[df['Year'] != year]

def get_list_combinations(features, r=-1):
    '''
    Returns a list of all possible combinations of a list of features

    features : list[]
    r : int, possible number of features per combination.
            If -1, uses length of features list
    '''
    feature_count = len(features) if r == -1 else r
    list_combinations = []
    for n in range(feature_count + 1):
        list_combinations += list(combinations(features, n))
    return list_combinations

def ols_regression_model(df, X_cols, y_col):
    '''
    Fit the data with OLS regression and return a model containing the result

    df : pandas.DataFrame
    X_cols : list[str], each entry should match to a column in df
    y_col : str, should match to a column in df
    '''
    X = df[X_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def plot_elbow_method_k_means(df, cols, max_k):
    '''
    Calculates within-cluster-sum of squared errors (WSS) for different values of k
    
    df : pandas.DataFrame
    cols : list[str], each entry should match to a column in df
    max_k : int, the maximum k to test for k-means clusters, not inclusive
    '''
    k_range = range(1, max_k)
    wss = [cluster.KMeans(n_clusters=k, init='k-means++').fit(df[cols]).inertia_ for k in k_range]
    mycenters = pd.DataFrame({'Clusters': k_range, 'Within-Cluster-Sum of Squared Errors': wss})
    sns.lineplot(
        x='Clusters', 
        y='Within-Cluster-Sum of Squared Errors', 
        data=mycenters, 
        marker='*'
    )

def get_silhouette_score_df(df, cols, max_k):
    '''
    Return a pandas DataFrame of the silhouette scores 
    of a dataset for a range of values of k for k-means clustering.

    A score of 1 is the best possible, while a score of -1 is the worst possible

    df : pandas.DataFrame
    cols : list[str], each entry should match to a column in df
    max_k : int, the maximum k to test for k-means clusters, not inclusive
            must be 4 or greater
    '''
    silhouettes = pd.DataFrame({"k": range(3, max_k)})
    silhouettes["Silhouette Score for k(clusters)"] = silhouettes.apply(
        lambda row: metrics.silhouette_score(
            df[cols], 
            cluster.KMeans(
                n_clusters=row["k"], 
                init='k-means++',
                random_state=200
            ).fit(df[cols]).labels_, 
            metric='euclidean', 
            sample_size=1000, 
            random_state=200
        ), axis=1
    )
    return silhouettes.set_index("k")

def get_k_means_labels(df, cols, k):
    '''
    Return an array of labels that match to a cluster within a DataFrame.
    Each index of the array corresponds to the index in the DataFrame.

    df : pandas.DataFrame
    cols : list[str], each entry should match to a column in df
    k : int, the number of clusters to use
    '''
    kmeans = cluster.KMeans(n_clusters=k, init='k-means++')
    kmeans = kmeans.fit(df[cols])
    kmeans.cluster_centers_
    return kmeans.labels_