import pandas as pd
from pathlib import Path  

def main():
    df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption = read_original_datasets()
    df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption = cleanup_original_datasets(df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption)
    df_merged = merge_datasets(df_cancer, df_milk, df_soy)
    df_new_merged = merge_new_datasets(df_life_expectancy, df_meat_consumption)
    save_merged_to_csv(df_new_merged)
    "save_merged_to_csv(df_merged)"

def read_original_datasets():
    df_cancer = pd.read_csv("original_datasets/share-of-population-with-cancer-types.csv").fillna("0")
    df_milk = pd.read_csv("original_datasets/per-capita-milk-consumption.csv").fillna("0")
    df_soy = pd.read_csv("original_datasets/soybean-production-and-use.csv").fillna("0")

    df_life_expectancy = pd.read_csv("original_datasets/life-expectancy.csv").fillna("0")
    df_meat_consumption = pd.read_csv("original_datasets/per-capita-meat-consumption-by-type-kilograms-per-year.csv").fillna("0")
    return df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption

def cleanup_original_datasets(df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption):
    # Rename entries
    df_milk["Entity"] = df_milk["Entity"].replace("Sudan (former)", "Sudan")
    df_meat_consumption["Entity"] = df_meat_consumption["Entity"].replace("Sudan (former)", "Sudan")
    # Rename and prune columns
    df_life_expectancy = df_life_expectancy.loc[df_life_expectancy["Entity"] != "World"].rename(columns={
        "Life expectancy": "Life Expectancy"
    })[["Entity", "Year", "Life Expectancy"]]
    df_meat_consumption = df_meat_consumption.loc[df_meat_consumption["Entity"] != "World"].rename(columns={
        "Meat, Other, Food supply quantity (kg/capita/yr) (FAO, 2020)": "Other Meats Consumption",
        "Mutton & Goat meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Mutton and Goat Consumption",
        "Bovine meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Beef and Buffallo Consumption",
        "Pigmeat food supply quantity (kg/capita/yr) (FAO, 2020)": "Pigmeat Consumption",
        "Poultry meat food supply quantity (kg/capita/yr) (FAO, 2020)": "Poultry Consumption"
    })[["Entity", "Year", "Mutton and Goat Consumption", "Beef and Buffallo Consumption", "Pigmeat Consumption",
    "Poultry Consumption", "Other Meats Consumption"]]
    df_cancer = df_cancer.loc[df_cancer["Entity"] != "World"].rename(columns={
        "Prevalence - Breast cancer - Sex: Both - Age: Age-standardized (Percent)": "Breast Cancer Rates",
    })[["Entity", "Year", "Breast Cancer Rates"]]
    df_milk = df_milk.loc[df_milk["Entity"] != "World"].rename(columns={
        "Milk - Excluding Butter - Food supply quantity (kg/capita/yr) (FAO, 2020)": "Milk Consumption"
    })[["Entity", "Year", "Milk Consumption"]]
    df_soy = df_soy.loc[df_soy["Entity"] != "World"].rename(columns={
        "Soybeans | 00002555 || Food | 005142 || tonnes": "Soy Consumption"
    })[["Entity", "Year", "Soy Consumption"]]
    return df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption

def merge_datasets(df_cancer, df_milk, df_soy):
    return df_cancer[[
        "Entity", "Year", "Breast Cancer Rates"
    ]].merge(
        df_milk,
        on=["Entity", "Year"]
    ).merge(
        df_soy,
        on=["Entity", "Year"]
    )

def merge_new_datasets(df_life_expectancy, df_meat_consumption):
    return df_life_expectancy[[
        "Entity", "Year", "Life Expectancy"
    ]].merge(
        df_meat_consumption,
        on=["Entity", "Year"]
    )

def save_merged_to_csv(df_merged):
    filepath = Path('updated_datasets/merged_dataset.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_merged.to_csv(filepath)

main()