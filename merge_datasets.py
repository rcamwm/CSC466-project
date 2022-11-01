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

# %% 
def get_life_expectancy_df():
    # Import CSV as Pandas DataFrame
    df_life_expectancy = pd.read_csv("original_datasets/life-expectancy.csv").fillna("0")

    # Rename and prune columns
    df_life_expectancy = df_life_expectancy.loc[df_life_expectancy["Entity"] != "World"].rename(columns={
        "Life expectancy": "Life Expectancy"
    })[["Entity", "Year", "Life Expectancy"]]

    return df_life_expectancy

# %% 
def get_world_gdp_df():
    # Import CSV as Pandas DataFrame
    df_world_gdp = pd.read_csv("original_datasets/country-regional-world-gdp.csv").fillna("0")

    return df_world_gdp

# %% 
def merge_datasets(df_life_expectancy, df_meat_consumption, df_world_gdp):
    return df_life_expectancy[[
        "Entity", "Year", "Life Expectancy"
    ]].merge(
        df_meat_consumption,
        on=["Entity", "Year"]
    ).merge(
        df_world_gdp,
        on=["Entity", "Year"]
    )
    return df_one

# %% 
def save_merged_to_csv(df_merged):
    filepath = Path('updated_datasets/merged_dataset.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_merged.to_csv(filepath)

# %%
save_merged_to_csv(
    merge_datasets(
        get_life_expectancy_df(),
        get_meat_consumption_df(),
        get_world_gdp_df()
    )
)

# %%
