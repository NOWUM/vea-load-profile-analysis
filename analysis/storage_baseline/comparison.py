import os

import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
import numpy as np

from dotenv import load_dotenv
load_dotenv()


def compare(uri: str):
    # get master data
    master = pd.read_sql("SELECT * FROM vea_industrial_load_profiles.master", uri)
    master.set_index("id", inplace=True)
    master.sort_index(inplace=True)

    # in the authors database, the following columns are already present in master
    # this will lead to problems later on, so we drop them here
    # if you have created your own features by the "create_master_features.py", you
    # do not need to worry about it
    cols_to_drop = ["energy_costs_eur", "capacity_costs_eur", "total_costs_eur"]
    for col_to_drop in cols_to_drop:
        try:
            master.drop(columns=col_to_drop, inplace=True)
        except KeyError as e:
            continue

    print(master.head())

    # get baseline data
    baseline = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%base%%'", uri)
    baseline["id"] = baseline["name"].str.split("_").str[0].astype(int)
    baseline.set_index("id", inplace=True)
    baseline.sort_index(inplace=True)
    print(baseline.head())

    # get storage scenario data
    storage = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%storage_only'", uri)
    storage["id"] = storage["name"].str.split("_").str[0].astype(int)
    storage.set_index("id", inplace=True)
    storage.sort_index(inplace=True)
    print(storage.head())

    print("")
    print("################################")
    print("#       general analysis       #")
    print("################################")
    total_profiles_analyzed = len(storage)
    profiles_using_storage = storage[storage["storage_invest_eur"] > 0]
    n_profiles_using_storage = len(profiles_using_storage)
    perc_profiles_using_storage = (n_profiles_using_storage / total_profiles_analyzed) * 100

    print(f"{total_profiles_analyzed=}")
    print(f"{n_profiles_using_storage=}")
    print(f"Percentage of profiles using storage: {perc_profiles_using_storage:.2f} %")

    print("")
    print("################################")
    print("#    battery system sizes      #")
    print("################################")
    stor_cap_col = "storage_capacity_kwh"
    median_storage_size = profiles_using_storage[stor_cap_col].median()
    print(f"{median_storage_size=:.4f} kWh")
    mean_storage_size = profiles_using_storage[stor_cap_col].mean()
    print(f"{mean_storage_size=:.4f} kWh")
    min_storage_size = profiles_using_storage[stor_cap_col].min()
    print(f"{min_storage_size=:.4f} kWh")
    max_storage_size = profiles_using_storage[stor_cap_col].max()
    print(f"{max_storage_size=:.4f} kWh")
    q95_storage_size = profiles_using_storage[stor_cap_col].quantile(0.95)
    print(f"{q95_storage_size=:.2f} kWh")

    print("----------------------------------")
    inv_cap_col = "inverter_capacity_kw"
    median_inverter_size = profiles_using_storage[inv_cap_col].median()
    print(f"{median_inverter_size=:.2f} kW")
    mean_inverter_size = profiles_using_storage[inv_cap_col].mean()
    print(f"{mean_inverter_size=:.2f} kW")
    min_inverter_size = profiles_using_storage[inv_cap_col].min()
    print(f"{min_inverter_size=:.2f} kW")
    max_inverter_size = profiles_using_storage[inv_cap_col].max()
    print(f"{max_inverter_size=:.2f} kW")
    q95_inverter_size = profiles_using_storage[inv_cap_col].quantile(0.95)
    print(f"{q95_inverter_size=:.2f} kW")

    fig_df = profiles_using_storage.copy()
    fig_df = fig_df[fig_df[stor_cap_col] <= q95_storage_size]
    fig_df = fig_df[fig_df[inv_cap_col] <= q95_inverter_size]
    fig_df = fig_df.rename(columns={stor_cap_col: "Storage", inv_cap_col: "Inverter"})
    fig = px.box(
        data_frame=fig_df,
        x=["Inverter", "Storage"],
        title="Battery system sizes for 95%-quantile")
    fig.update_layout(xaxis_title="Capacity in kWh (storage) / kW (inverter)", yaxis_title="")
    fig.show()

    print("")
    print("################################")
    print("#  battery system investments  #")
    print("################################")
    stor_inv_col = "storage_invest_eur"
    median_storage_invest = profiles_using_storage[stor_inv_col].median()
    print(f"{median_storage_invest=:.2f} €")
    mean_storage_invest = profiles_using_storage[stor_inv_col].mean()
    print(f"{mean_storage_invest=:.2f} €")
    min_storage_invest = profiles_using_storage[stor_inv_col].min()
    print(f"{min_storage_invest=:.2f} €")
    max_storage_invest = profiles_using_storage[stor_inv_col].max()
    print(f"{max_storage_invest=:.2f} €")
    q95_storage_invest = profiles_using_storage[stor_inv_col].quantile(0.95)
    print(f"{q95_storage_invest=:.2f} €")

    print("----------------------------------")
    inv_invest_col = "inverter_invest_eur"
    median_inverter_invest = profiles_using_storage[inv_invest_col].median()
    print(f"{median_inverter_invest=:.2f} €")
    mean_inverter_invest = profiles_using_storage[inv_invest_col].mean()
    print(f"{mean_inverter_invest=:.2f} €")
    min_inverter_invest = profiles_using_storage[inv_invest_col].min()
    print(f"{min_inverter_invest=:.2f} €")
    max_inverter_invest = profiles_using_storage[inv_invest_col].max()
    print(f"{max_inverter_invest=:.2f} €")
    q95_inverter_invest = profiles_using_storage[inv_invest_col].quantile(0.95)
    print(f"{q95_inverter_invest=:.2f} €")

    fig_df = profiles_using_storage.copy()
    fig_df = fig_df[fig_df[stor_inv_col] <= q95_storage_invest]
    fig_df = fig_df[fig_df[inv_invest_col] <= q95_inverter_invest]
    fig_df = fig_df.rename(columns={stor_inv_col: "Storage", inv_invest_col: "Inverter"})
    fig = px.box(
        data_frame=fig_df,
        x=["Inverter", "Storage"],
        title="Battery system investments for 95%-quantile")
    fig.update_layout(xaxis_title="Storage system investments in €", yaxis_title="")
    fig.show()

    # calculate difference between both scenarios
    abs_diff = baseline.drop(columns="name") - storage.drop(columns="name")

    rel_diff = (baseline.drop(columns="name") - storage.drop(columns="name")) / baseline.drop(columns="name")
    rel_diff.dropna(subset="total_yearly_costs_eur", inplace=True)

    print("")
    print("################################")
    print("#     total yearly savings     #")
    print("################################")
    tot_y_sav_col = "total_yearly_costs_eur"
    median_total_yearly_savings = abs_diff[tot_y_sav_col].median()
    print(f"{median_total_yearly_savings=:.2f} €")
    mean_total_yearly_savings = abs_diff[tot_y_sav_col].mean()
    print(f"{mean_total_yearly_savings=:.2f} €")
    min_total_yearly_savings = abs_diff[tot_y_sav_col].min()
    print(f"{min_total_yearly_savings=:.2f} €")
    max_total_yearly_savings = abs_diff[tot_y_sav_col].max()
    print(f"{max_total_yearly_savings=:.2f} €")
    q95_total_yearly_savings = abs_diff[tot_y_sav_col].quantile(0.95)
    print(f"{q95_total_yearly_savings=:.2f} €")

    fig_df = abs_diff.copy()
    fig_df = fig_df[fig_df[tot_y_sav_col] <= q95_total_yearly_savings]
    fig_df = fig_df.rename(columns={tot_y_sav_col: "Savings"})
    fig = px.box(
        data_frame=fig_df,
        x=["Savings"],
        title="Total yearly savings for 95%-quantile")
    fig.update_layout(xaxis_title="Total yearly savings in €", yaxis_title="")
    fig.show()

    print("")
    print("################################")
    print("# percentual yearly savings    #")
    print("################################")
    perc_yearly_savings = (abs_diff["total_yearly_costs_eur"] / baseline["total_yearly_costs_eur"]) * 100
    median_perc_yearly_savings = perc_yearly_savings.median()
    print(f"{median_perc_yearly_savings=:.2f} %")
    mean_perc_yearly_savings = perc_yearly_savings.mean()
    print(f"{mean_perc_yearly_savings=:.2f} %")
    min_perc_yearly_savings = perc_yearly_savings.min()
    print(f"{min_perc_yearly_savings=:.2f} %")
    max_perc_yearly_savings = perc_yearly_savings.max()
    print(f"{max_perc_yearly_savings=:.2f} %")
    q95_perc_yearly_savings = perc_yearly_savings.quantile(0.8)
    print(f"{q95_perc_yearly_savings=:.2f} %")
    print("-----------------------")
    n_profiles_1pct_yearly_savings = len(perc_yearly_savings[perc_yearly_savings > 1])
    print(f"Number of profiles with more than 1% yearly savings: {n_profiles_1pct_yearly_savings}")
    perc_profiles_1pct_yearly_savings = (n_profiles_1pct_yearly_savings / len(perc_yearly_savings)) * 100
    print(f"Percentage of profiles with more than 1% yearly savings: {perc_profiles_1pct_yearly_savings:.2f} %")
    n_profiles_2pct_yearly_savings = len(perc_yearly_savings[perc_yearly_savings > 2])
    print(f"Number of profiles with more than 2% yearly savings: {n_profiles_2pct_yearly_savings}")
    perc_profiles_2pct_yearly_savings = (n_profiles_2pct_yearly_savings / len(perc_yearly_savings)) * 100
    print(f"Percentage of profiles with more than 2% yearly savings: {perc_profiles_2pct_yearly_savings:.2f} %")
    n_profiles_3pct_yearly_savings = len(perc_yearly_savings[perc_yearly_savings > 3])
    print(f"Number of profiles with more than 3% yearly savings: {n_profiles_3pct_yearly_savings}")
    perc_profiles_3pct_yearly_savings = (n_profiles_3pct_yearly_savings / len(perc_yearly_savings)) * 100
    print(f"Percentage of profiles with more than 3% yearly savings: {perc_profiles_3pct_yearly_savings:.2f} %")



    fig_df = pd.DataFrame()
    fig_df["Savings"] = perc_yearly_savings.copy()
    fig_df = fig_df[fig_df["Savings"] <= q95_perc_yearly_savings]
    fig_df = fig_df.rename(columns={"Savings": "Savings"})
    fig = px.box(
        data_frame=fig_df,
        x="Savings",
        title="Relative yearly savings for 95%-quantile")
    fig.update_layout(xaxis_title="Relative yearly savings in %", yaxis_title="")
    fig.show()

    print("")
    print("#####################################")
    print("# total grid capacity costs savings #")
    print("#####################################")
    yearly_cap_cost_sav_col = "grid_capacity_costs_eur"
    median_yearly_cap_cost_savings = abs_diff[yearly_cap_cost_sav_col].median()
    print(f"{median_yearly_cap_cost_savings=:.2f} €")
    mean_yearly_cap_cost_savings = abs_diff[yearly_cap_cost_sav_col].mean()
    print(f"{mean_yearly_cap_cost_savings=:.2f} €")
    min_yearly_cap_cost_savings = abs_diff[yearly_cap_cost_sav_col].min()
    print(f"{min_yearly_cap_cost_savings=:.2f} €")
    max_yearly_cap_cost_savings = abs_diff[yearly_cap_cost_sav_col].max()
    print(f"{max_yearly_cap_cost_savings=:.2f} €")
    q95_yearly_cap_cost_savings = abs_diff[yearly_cap_cost_sav_col].quantile(0.95)
    print(f"{q95_yearly_cap_cost_savings=:.2f} €")

    fig_df = abs_diff.copy()
    fig_df = fig_df[fig_df[yearly_cap_cost_sav_col] <= q95_yearly_cap_cost_savings]
    fig_df = fig_df.rename(columns={yearly_cap_cost_sav_col: "Savings"})
    fig = px.box(
        data_frame=fig_df,
        x="Savings",
        title="Yearly capacity costs savings for 95%-quantile")
    fig.update_layout(xaxis_title="Total yearly capacity costs savings in €", yaxis_title="")
    fig.show()

    print("")
    print("########################################")
    print("# relative grid capacity costs savings #")
    print("########################################")
    perc_yearly_cap_cost_savings = (abs_diff["grid_capacity_costs_eur"] / baseline["grid_capacity_costs_eur"]) * 100
    median_perc_yearly_cap_cost_savings = perc_yearly_cap_cost_savings.median()
    print(f"{median_perc_yearly_cap_cost_savings=:.2f} %")
    mean_perc_yearly_cap_cost_savings = perc_yearly_cap_cost_savings.mean()
    print(f"{mean_perc_yearly_cap_cost_savings=:.2f} %")
    min_perc_yearly_cap_cost_savings = perc_yearly_cap_cost_savings.min()
    print(f"{min_perc_yearly_cap_cost_savings=:.2f} %")
    max_perc_yearly_cap_cost_savings = perc_yearly_cap_cost_savings.max()
    print(f"{max_perc_yearly_cap_cost_savings=:.2f} %")
    q95_perc_yearly_cap_cost_savings = perc_yearly_cap_cost_savings.quantile(0.8)
    print(f"{q95_perc_yearly_cap_cost_savings=:.2f} %")
    print("-----------------------")
    n_profiles_1pct_yearly_cap_cost_savings = len(perc_yearly_cap_cost_savings[perc_yearly_cap_cost_savings > 1])
    print(f"Number of profiles with more than 1% yearly savings: {n_profiles_1pct_yearly_cap_cost_savings}")
    perc_profiles_1pct_yearly_cap_cost_savings = (len(perc_yearly_cap_cost_savings[perc_yearly_cap_cost_savings > 1]) / len(perc_yearly_savings)) * 100
    print(f"Percentage of profiles with more than 1% yearly savings: {perc_profiles_1pct_yearly_cap_cost_savings:.2f} %")
    n_profiles_2pct_yearly_cap_cost_savings = len(perc_yearly_cap_cost_savings[perc_yearly_cap_cost_savings > 2])
    print(f"Number of profiles with more than 2% yearly savings: {n_profiles_2pct_yearly_cap_cost_savings}")
    perc_profiles_2pct_yearly_cap_cost_savings = (len(perc_yearly_cap_cost_savings[perc_yearly_cap_cost_savings > 2]) / len(perc_yearly_savings)) * 100
    print(f"Percentage of profiles with more than 2% yearly savings: {perc_profiles_2pct_yearly_cap_cost_savings:.2f} %")

    fig_df = pd.DataFrame()
    fig_df["Savings"] = perc_yearly_cap_cost_savings.copy()
    fig_df = fig_df[fig_df["Savings"] <= q95_perc_yearly_cap_cost_savings]
    fig = px.box(
        data_frame=fig_df,
        x="Savings",
        title="Relative yearly capacity cost savings for 95%-quantile")
    fig.update_layout(xaxis_title="Relative yearly capacity cost savings in %", yaxis_title="")
    fig.show()

    print("")
    print("#################################")
    print("# absolute correlation analysis #")
    print("#################################")
    # merge savings onto master (with features)
    abs_diff_with_master = pd.merge(left=abs_diff, right=master, how="left", left_index=True, right_index=True)
    print(abs_diff_with_master.head())

    # add some more features
    abs_diff_with_master["std_by_mean"] = abs_diff_with_master["std_kw"] / abs_diff_with_master["mean_load_kw"]
    abs_diff_with_master["std_by_peak"] = abs_diff_with_master["std_kw"] / abs_diff_with_master["peak_load_kw"]
    abs_diff_with_master["peak_by_mean"] = abs_diff_with_master["peak_load_kw"] / abs_diff_with_master["mean_load_kw"]

    # generate correlation df
    cols_to_drop = [
        "grid_level",
        "zip_code",
        "sector_group_id",
        "sector_group",
        "solar_invest_eur",
        "solar_annuity_eur",
        "solar_capacity_kwp"]
    abs_correlations_df = abs_diff_with_master.drop(columns=cols_to_drop).corr()
    fig = px.imshow(abs_correlations_df, title="Correlation coefficients")
    fig.show()

    # show correlation coefficients
    fig_df = abs_correlations_df[["total_yearly_costs_eur"]].round(2)
    fig_df.sort_values("total_yearly_costs_eur", inplace=True, ascending=False)
    fig = px.bar(
        data_frame=fig_df,
        y="total_yearly_costs_eur",
        text_auto=True,
        title="Correlation between different load profile characteristics and total yearly savings")
    fig.update_layout(yaxis_title="Correlation coefficients", xaxis_title="Variable")
    fig.show()

    # table showing absolute correlations
    df = pd.DataFrame()
    i = 0
    for var in abs_correlations_df.index:#[correlations_df["total_yearly_costs_eur"] > 0.3].index:
        corr, p_value = pearsonr(y=abs_diff_with_master.dropna()["total_yearly_costs_eur"], x=abs_diff_with_master.dropna()[var])
        df.loc[i, "var"] = var
        df.loc[i, "corr"] = corr
        df.loc[i, "p_value"] = p_value
        i += 1
    print(df.sort_values("corr", ascending=False, ignore_index=True))


    print("")
    print("#################################")
    print("# relative correlation analysis #")
    print("#################################")
    # merge savings onto master (with features)
    rel_diff_with_master = pd.merge(left=rel_diff, right=master, how="left", left_index=True, right_index=True)
    rel_diff_with_master.head()

    # add some more features
    rel_diff_with_master["std_by_mean"] = rel_diff_with_master["std_kw"] / rel_diff_with_master["mean_load_kw"]
    rel_diff_with_master["std_by_peak"] = rel_diff_with_master["std_kw"] / rel_diff_with_master["peak_load_kw"]
    rel_diff_with_master["peak_by_mean"] = rel_diff_with_master["peak_load_kw"] / rel_diff_with_master["mean_load_kw"]

    # matrix plot showing relative correlations
    cols_to_drop = [
        "grid_level",
        "zip_code",
        "sector_group_id",
        "sector_group",
        "solar_invest_eur",
        "solar_annuity_eur",
        "solar_capacity_kwp"]
    rel_correlations_df = rel_diff_with_master.drop(columns=cols_to_drop).corr()
    fig = px.imshow(rel_correlations_df, title="Correlation coefficients for relative yearly savings")
    fig.show()

    # bar plot showing relative correlations
    fig_df = rel_correlations_df[["total_yearly_costs_eur"]].round(2)
    fig_df.sort_values("total_yearly_costs_eur", inplace=True, ascending=False)
    fig_df.dropna(inplace=True)
    fig = px.bar(
        data_frame=fig_df,
        y="total_yearly_costs_eur",
        text_auto=True,
        title="Correlation between different load profile characteristics and relative yearly savings")
    fig.update_layout(yaxis_title="Correlation coefficient", xaxis_title="Variable")

    # table showing relative correlations
    df = pd.DataFrame()
    i = 0
    for var in rel_correlations_df.index:
        if np.inf in rel_diff_with_master[var]:
            continue
        elif -np.inf in rel_diff_with_master[var]:
            continue
        elif np.nan in rel_diff_with_master[var]:
            continue
        elif rel_diff_with_master[var].isin([np.nan, np.inf, -np.inf]).any():
            continue
        if "storage" in var or "inverter" in var:
            continue

        corr, p_value = pearsonr(y=rel_diff_with_master["total_yearly_costs_eur"], x=rel_diff_with_master[var])
        df.loc[i, "var"] = var
        df.loc[i, "corr"] = corr
        df.loc[i, "p_value"] = p_value
        i += 1

    print(df.sort_values("corr", ascending=False, ignore_index=True))


if __name__ == "__main__":

    # load DB URI
    uri = os.getenv("DB_URI")

    compare(uri=uri)