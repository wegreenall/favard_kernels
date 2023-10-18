"""
This script will take an initial look at the dataset on pitstops,
to see whether it is reasonable to use the data as a discrete variable.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def clean_outliers(df, column_name):
    """
    Remove outliers from a dataframe, based on a column name.
    """
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    df = df.loc[
        (df[column_name] > (q1 - 1.5 * iqr))
        & (df[column_name] < (q3 + 1.5 * iqr))
    ]
    return df


# Import libraries
print(os.listdir())
filename = "./data/pit_stops.csv"
df = pd.read_csv(filename)

cleaned_df = clean_outliers(df, "milliseconds")
plt.hist(cleaned_df["milliseconds"], bins=100)
print(cleaned_df["milliseconds"].describe())
plt.show()

# with the cleaned outliers millisecond data, get the values of another random
# variable: which lap it was on. Is there any correlation between lap and pit
# stop count? If yes, it would imply that crews get
# better at pit stops even over the course of a race.
# The implication is that pre-race practice could be an effective tool for
# improving pit stop times.
show_cleaned = True
if show_cleaned:
    plt.scatter(
        cleaned_df["stop"],
        cleaned_df["milliseconds"],
        marker="x",
        linewidths=0.1,
    )
    plt.show()
else:
    plt.scatter(df["stop"], df["milliseconds"], marker="x", linewidths=0.1)
    plt.show()

# can we use the lap number as a discrete variable?
input_sample, output_sample = get_data(cleaned_df, "stop", "milliseconds")
