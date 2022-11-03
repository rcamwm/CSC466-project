# %%
import pandas as pd
from pathlib import Path

# %%
def get_meat_consumption_df():
    # Import CSV as Pandas DataFrame
    df_meat_consumption = pd.read_csv(
        "original_datasets/per-capita-meat-consumption-by-type-kilograms-per-year.csv"
    ).fillna("0")

    # Rename entries
    df_meat_consumption["Entity"] = df_meat_consumption["Entity"].replace("Sudan (former)", "Sudan")

    # Rename and prune columns
    df_meat_consumption = df_meat_consumption.loc[df_meat_consumption["Entity"] != "World"].rename(columns={
        "Meat, Other, Food supply quantity (kg/capita/yr) (FAO, 2020)": "Other Meats Consumption",
        "Mutton & Goat meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Mutton and Goat Consumption",
        "Bovine meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Beef and Buffallo Consumption",
        "Pigmeat food supply quantity (kg/capita/yr) (FAO, 2020)": "Pigmeat Consumption",
        "Poultry meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Poultry Consumption"
    })[["Entity", "Year", "Mutton and Goat Consumption", "Beef and Buffallo Consumption", "Pigmeat Consumption",
    "Poultry Consumption", "Other Meats Consumption"]]

    return df_meat_consumption

def get_life_expectancy_df():
    # Import CSV as Pandas DataFrame
    df_life_expectancy = pd.read_csv("original_datasets/life-expectancy.csv").fillna("0")

    # Rename and prune columns
    df_life_expectancy = df_life_expectancy.loc[df_life_expectancy["Entity"] != "World"].rename(columns={
        "Life expectancy": "Life Expectancy"
    })[["Entity", "Year", "Life Expectancy"]]

    return df_life_expectancy

def get_child_mortality_df():
    # Import CSV as Pandas DataFrame
    df_child_mortality = pd.read_csv("original_datasets/child-mortality-by-income-level-of-country.csv").fillna("0")

    return df_child_mortality

def get_gdp_per_capita_df():
    # Import CSV as Pandas DataFrame
    df_gdp_per_capita = pd.read_csv("original_datasets/gdp-per-capita-worldbank.csv").fillna("0")

    return df_gdp_per_capita

# %%
def merge_datasets(base_df, added_df):
    return base_df.merge(
        added_df,
        on=["Entity", "Year"],
    )

def save_merged_to_csv(df_merged):
    filepath = Path('updated_datasets/merged_dataset.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_merged.to_csv(filepath)

# %%
df_merged = pd.DataFrame(dtype="float64")
dfs = [
    get_life_expectancy_df(),
    get_meat_consumption_df(),
    get_child_mortality_df(),
    get_gdp_per_capita_df(),
]
for df in dfs:
    df_merged = df if df_merged.empty else merge_datasets(
        df_merged, df.loc[:, df.columns != 'Code']
    )

save_merged_to_csv(df_merged)
df_merged

# %%
