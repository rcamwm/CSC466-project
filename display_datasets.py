# %%
import pandas as pd

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

#%% 
df = pd.read_csv("updated_datasets/merged_dataset.csv")
# %%
show_milk_breastcancer_graph(df, 2015)

# %%
show_soy_breastcancer_graph(df, 2015)
# %%
