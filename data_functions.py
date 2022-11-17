import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.cluster as cluster

def ols_regression_model_summary(df, X_cols, y_col):
    '''
    Fit the data with OLS regression and return a table containing the result

    df : pandas.DataFrame
    X_cols : list[str], each entry should match to a column in df
    y_col : str, should match to a column in df
    '''
    X = df[X_cols]
    y = df[y_col]
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def plot_elbow_method_k_means(df, max_k):
    '''
    Calculates within-cluster-sum of squared errors (WSS) for different values of k
    
    df : pandas.DataFrame
    max_k : int, the maximum k to test for k-means clusters, not inclusive
    '''
    k_range = range(1, max_k)
    wss = [cluster.KMeans(n_clusters=k, init='k-means++').fit(df).inertia_ for k in k_range]
    mycenters = pd.DataFrame({'Clusters': k_range, 'Within-Cluster-Sum of Squared Errors': wss})
    sns.lineplot(
        x='Clusters', 
        y='Within-Cluster-Sum of Squared Errors', 
        data=mycenters, 
        marker='*'
    )

def get_silhouette_score_df(df, max_k):
    '''
    Return a pandas DataFrame of the silhouette scores 
    of a dataset for a range of values of k for k-means clustering.

    A score of 1 is the best possible, while a score of -1 is the worst possible

    df : pandas.DataFrame
    max_k : int, the maximum k to test for k-means clusters, not inclusive
            must be 4 or greater
    '''
    
    silhouettes = pd.DataFrame({"k": range(3, max_k)})
    silhouettes["Silhouette Score for k(clusters)"] = silhouettes.apply(
        lambda row: metrics.silhouette_score(
            df, 
            cluster.KMeans(
                n_clusters=row["k"], 
                init='k-means++',
                random_state=200
            ).fit(df).labels_, 
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