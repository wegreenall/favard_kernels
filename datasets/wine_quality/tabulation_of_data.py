import pickle
import tabulate
import math
from statistics import mean, stdev as std

# get the data
red_wine_data = pickle.load(open("red_wine_pd.p", "rb"))
white_wine_data = pickle.load(open("white_wine_pd.p", "rb"))
total_wine_data = pickle.load(open("total_wine_pd.p", "rb"))
red_wine_mercer_data = pickle.load(open("red_wine_mercer_pd.p", "rb"))
white_wine_mercer_data = pickle.load(open("white_wine_mercer_pd.p", "rb"))
total_wine_mercer_data = pickle.load(open("total_wine_mercer_pd.p", "rb"))

# get the data
red_wine_diffs = [
    (favard - mercer).item()
    for favard, mercer in zip(red_wine_data, red_wine_mercer_data)
]
red_wine_mean_diff = mean(red_wine_diffs)
red_wine_std_diff = std(red_wine_diffs)
white_wine_diffs = [
    (favard - mercer).item()
    for favard, mercer in zip(white_wine_data, white_wine_mercer_data)
]
white_wine_mean_diff = mean(white_wine_diffs)
white_wine_std_diff = std(white_wine_diffs)
total_wine_diffs = [
    (favard - mercer).item()
    for favard, mercer in zip(total_wine_data, total_wine_mercer_data)
]
total_wine_mean_diff = mean(total_wine_diffs)
total_wine_std_diff = std(total_wine_diffs)
# breakpoint()
titles = ["Red Wine", "White Wine", "Total Wine"]
means = [red_wine_mean_diff, white_wine_mean_diff, total_wine_mean_diff]
stds = [red_wine_std_diff, white_wine_std_diff, total_wine_std_diff]
wine_table = tabulate.tabulate(
    list(zip(titles, means, stds)),
    headers=("Case", "Means", "Standard Deviation"),
    tablefmt="simple",
)
print(wine_table)

file = open("wine_table.mk", "w")
file.write(wine_table)
file.close()
