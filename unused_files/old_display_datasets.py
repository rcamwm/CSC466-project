import pandas as pd

# %%
def show_milk_breastcancer_graph(df, year):
    '''
    Display the scatter plot: Milk Comsumption (x-axis) vs Breast Cancer Rate (y-axis)

            Parameters:
                    df : pandas dataframe containing the info
                    b (int): desired year to display the info
    '''
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Milk Consumption",
        y="Breast Cancer Rates",
        title="Breast Cancer Rates in {}".format(year)
    )
    
def show_soy_breastcancer_graph(df, year):
    '''
    Display the scatter plot: Soy Comsumption (x-axis) vs Breast Cancer Rate (y-axis)

            Parameters:
                    df : pandas dataframe containing the info
                    b (int): desired year to display the info
    '''
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Soy Consumption",
        y="Breast Cancer Rates",
        title="Breast Cancer Rates in {}".format(year)
    )

def show_mutton_and_goat_graph(df, year):
    '''
    Display the scatter plot: Mutton and Goat Comsumption (x-axis) vs Life Expectancy (y-axis)

            Parameters:
                    df : pandas dataframe containing the info
                    b (int): desired year to display the info
    '''
    df.loc[
        df["Year"] == year
    ].plot(
        kind="scatter",
        x="Mutton and Goat Consumption",
        y="Life Expectancy",
        title="Life Expectancy in {}".format(year)
    )

# %%
df = pd.DataFrame() # replace
class object1:
    def __init__(self, entity: str, year: int, gpa_per_capita: float) -> None:
        self.entity = entity
        self.year = year
        self.gpa_per_capita = gpa_per_capita
object1_list = []
df.apply(lambda row : object1_list.append(object1(row['Entity'], row['Year'], row['GDP per capita (2017 international $)'])), axis = 1)
