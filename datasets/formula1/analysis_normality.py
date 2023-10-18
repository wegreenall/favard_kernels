if __name__ == "__main__":
    datasets = [DataSet.RED, DataSet.WHITE, DataSet.BOTH]
    # perform a kolmogorov-smirnov test on the data
    stats = []
    p_values = []
    for dataset in datasets:
        _, data = get_data(dataset, standardise=True)
        # perform a kolmogorov-smirnov test on the data, with respect to the normal cdf
        ks_test = kstest(data, "norm")
        stats.append(ks_test[0])
        p_values.append(ks_test[1])
    print(stats)
    print(p_values)
    table = tt.Texttable()
    table.set_precision(10)
    table.set_cols_align(["c", "c", "c"])
    table.set_cols_valign(["m", "m", "m"])
    table.add_rows(
        [
            ["Dataset", "KS Statistic", "p-value"],
            ["Red Wine", stats[0], p_values[0]],
            ["White Wine", stats[1], p_values[1]],
            ["Amalgamated Dataset", stats[2], p_values[2]],
        ]
    )
    table_tex = latextable.draw_latex(
        table, caption="KS Test Results", label="tab:ks_test"
    )
    # write table tex to file
    with open("wine_dataset_ks.tex", "w") as f:
        f.write(table_tex)

    # print(ks_test)
