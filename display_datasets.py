# %%
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

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