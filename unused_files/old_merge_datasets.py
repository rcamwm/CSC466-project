import pandas as pd
from pathlib import Path  

def main():
    df_cancer, df_milk, df_soy = read_original_datasets()
    df_cancer, df_milk, df_soy = cleanup_original_datasets(df_cancer, df_milk, df_soy)
    df_merged = merge_datasets(df_cancer, df_milk, df_soy)
    save_merged_to_csv(df_merged)

def read_original_datasets():
    df_cancer = pd.read_csv("../original_datasets/share-of-population-with-cancer-types.csv").fillna("0")
    df_milk = pd.read_csv("../original_datasets/per-capita-milk-consumption.csv").fillna("0")
    df_soy = pd.read_csv("../original_datasets/soybean-production-and-use.csv").fillna("0")
    return df_cancer, df_milk, df_soy

def cleanup_original_datasets(df_cancer, df_milk, df_soy, df_life_expectancy, df_meat_consumption):
    # Rename entries
    df_milk["Entity"] = df_milk["Entity"].replace("Sudan (former)", "Sudan")

    # Rename and prune columns
    df_cancer = df_cancer.loc[df_cancer["Entity"] != "World"].rename(columns={
        "Prevalence - Breast cancer - Sex: Both - Age: Age-standardized (Percent)": "Breast Cancer Rates",
    })[["Entity", "Year", "Breast Cancer Rates"]]
    df_milk = df_milk.loc[df_milk["Entity"] != "World"].rename(columns={
        "Milk - Excluding Butter - Food supply quantity (kg/capita/yr) (FAO, 2020)": "Milk Consumption"
    })[["Entity", "Year", "Milk Consumption"]]
    df_soy = df_soy.loc[df_soy["Entity"] != "World"].rename(columns={
        "Soybeans | 00002555 || Food | 005142 || tonnes": "Soy Consumption"
    })[["Entity", "Year", "Soy Consumption"]]
    return df_cancer, df_milk, df_soy

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

def save_merged_to_csv(df_merged):
    filepath = Path('../updated_datasets/merged_dataset.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_merged.to_csv(filepath)

main()