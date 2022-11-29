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

def get_milk_consumption_df():
    # Import CSV as Pandas DataFrame
    df_milk_consumption = pd.read_csv(
        "original_datasets/per-capita-milk-consumption.csv"
    ).fillna("0")

    # Rename columns
    df_milk_consumption.columns = ["Entity", "Code", "Year", "Milk Consumption"]

    return df_milk_consumption

def get_alcohol_df():
    # Import CSV as Pandas DataFrame
    df_alcohol = pd.read_csv(
        "original_datasets/alcohol-consumption-rates.csv"
    ).fillna("0")

    # Rename columns
    df_alcohol.columns = ["Entity", "Code", "Year", "Alcohol Consumption"]

    return df_alcohol

def get_smoking_df(): 
    # Import CSV as Pandas DataFrame
    df_smoking = pd.read_csv(
        "original_datasets/smoking-rates.csv"
    ).fillna("0")

    # Rename columns
    df_smoking.columns = ["Entity", "Code", "Year", "Smoking Rates"]

    return df_smoking

def get_cancer_df():
    # Import CSV as Pandas DataFrame
    df_cancer = pd.read_csv(
        "original_datasets/share-of-population-with-cancer-types.csv"
    ).fillna("0")

    # Rename columns
    df_cancer.columns = [(col if " - " not in col else col.split(" - ")[1]) for col in df_cancer.columns]

    return df_cancer

def get_obesity_df():
    # Import CSV as Pandas DataFrame
    df_obesity = pd.read_csv(
        "original_datasets/share-of-adults-defined-as-obese.csv"
    ).fillna("0")

    df_obesity.columns = ["Entity", "Code", "Year", "Obesity Rate"]

    return df_obesity

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

def get_health_exp_df():
    # Import CSV as Pandas DataFrame
    df_health_exp = pd.read_csv("original_datasets/public-health-expenditure-share-GDP-OWID.csv").fillna("0")

    return df_health_exp

# %%
def merge_datasets(base_df, added_df):
    return base_df.merge(
        added_df,
        on=["Entity", "Year"],
    )

def save_merged_to_csv(df_merged):
    filepath = Path('updated_datasets/merged_dataset.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_merged.to_csv(filepath, index = False)

# %%
df_merged = pd.DataFrame(dtype="float64")
dfs = [
    get_life_expectancy_df(),
    get_meat_consumption_df(),
    get_milk_consumption_df(),
    get_cancer_df(),
    get_obesity_df(),
    get_child_mortality_df(),
    get_gdp_per_capita_df(),
    get_health_exp_df(),
]
for df in dfs:
    df_merged = df if df_merged.empty else merge_datasets(
        df_merged, df.loc[:, df.columns != 'Code']
    )

save_merged_to_csv(df_merged)
df_merged

# %%
