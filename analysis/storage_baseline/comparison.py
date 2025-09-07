import os
from pathlib import Path

import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import plotly.io as pio
pio.renderers.default = "browser"


def compare(uri: str, images_dir: str | Path):

    print("")
    print("################################")
    print("#            data              #")
    print("################################")
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

    print(master.head().to_markdown())

    # get baseline data
    baseline = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%base%%'", uri)
    baseline["id"] = baseline["name"].str.split("_").str[0].astype(int)
    baseline.set_index("id", inplace=True)
    baseline.sort_index(inplace=True)
    print(baseline.head().to_markdown())

    # get storage scenario data
    storage = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%storage_only'", uri)
    storage["id"] = storage["name"].str.split("_").str[0].astype(int)
    storage.set_index("id", inplace=True)
    storage.sort_index(inplace=True)
    print(storage.head().to_markdown())

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
    storage_size = profiles_using_storage["storage_capacity_kwh"].copy()
    storage_size.name = "Storage capacity in kWh"
    print(storage_size.describe().drop("count").to_markdown())

    print("")
    inverter_size = profiles_using_storage["inverter_capacity_kw"].copy()
    inverter_size.name = "Inverter capacity in kW"
    print(inverter_size.describe().drop("count").to_markdown())

    battery_size_fig_df = profiles_using_storage.copy()
    battery_size_fig_df = battery_size_fig_df.rename(columns={
        "storage_capacity_kwh": "Storage",
        "inverter_capacity_kw": "Inverter"})
    battery_size_fig = px.box(
        data_frame=battery_size_fig_df,
        x=["Inverter", "Storage"],
        title="Battery system sizes")
    battery_size_fig.update_layout(xaxis_title="Capacity in kWh (storage) / kW (inverter)", yaxis_title="")
    battery_size_fig.update_xaxes(range=[0, 200])
    battery_size_fig.write_image(f"{images_dir}/battery_size_box.pdf")
    battery_size_fig.show()

    print("")
    print("################################")
    print("#  battery system investments  #")
    print("################################")
    storage_invest = profiles_using_storage["storage_invest_eur"].copy()
    storage_invest.name = "Storage investment in eur"
    print(storage_invest.describe().drop("count").to_markdown())

    print("")
    inverter_invest = profiles_using_storage["inverter_invest_eur"].copy()
    inverter_invest.name = "Inverter investment in eur"
    print(inverter_invest.describe().drop("count").to_markdown())

    battery_investments_fig_df = profiles_using_storage.copy()
    battery_investments_fig_df = battery_investments_fig_df.rename(columns={
        "storage_invest_eur": "Storage",
        "inverter_invest_eur": "Inverter"})
    battery_investments_fig = px.box(
        data_frame=battery_investments_fig_df,
        x=["Inverter", "Storage"],
        title="Battery system investments")
    battery_investments_fig.update_layout(xaxis_title="Investments in €", yaxis_title="")
    battery_investments_fig.update_xaxes(range=[0, 50000])
    battery_investments_fig.write_image(f"{images_dir}/battery_investments_box.pdf")
    battery_investments_fig.show()

    print("")
    print("################################")
    print("#    battery system annuity    #")
    print("################################")
    abs_storage_invest = profiles_using_storage["storage_annuity_eur"].copy()
    abs_storage_invest.name = "Storage annuity in eur"
    print(abs_storage_invest.describe().drop("count").to_markdown())

    print("")
    abs_inverter_invest = profiles_using_storage["inverter_annuity_eur"].copy()
    abs_inverter_invest.name = "Inverter annuity in eur"
    print(abs_inverter_invest.describe().drop("count").to_markdown())

    battery_annuity_fig_df = profiles_using_storage.copy()
    battery_annuity_fig_df = battery_annuity_fig_df.rename(columns={
        "storage_annuity_eur": "Storage",
        "inverter_annuity_eur": "Inverter"})
    battery_annuity_fig = px.box(
        data_frame=battery_annuity_fig_df,
        x=["Inverter", "Storage"],
        title="Battery system annuity")
    battery_annuity_fig.update_layout(xaxis_title="Annuity in €", yaxis_title="")
    battery_annuity_fig.update_xaxes(range=[0, 3000])
    battery_annuity_fig.write_image(f"{images_dir}/battery_annuity_box.pdf")
    battery_annuity_fig.show()

    # calculate difference between both scenarios
    abs_diff = baseline.drop(columns="name") - storage.drop(columns="name")

    rel_diff = (baseline.drop(columns="name") - storage.drop(columns="name")) / baseline.drop(columns="name")
    rel_diff.dropna(subset="total_yearly_costs_eur", inplace=True)

    print("")
    print("################################")
    print("#     total yearly savings     #")
    print("################################")
    abs_yearly_savings = abs_diff["total_yearly_costs_eur"].copy()
    abs_yearly_savings.name = "Total yearly savings in Eur"
    print(abs_yearly_savings.describe().drop("count").to_markdown())

    total_yearly_savings_fig_df = abs_diff.copy()
    total_yearly_savings_fig_df = total_yearly_savings_fig_df.rename(columns={"total_yearly_costs_eur": "Savings"})
    total_yearly_savings_fig = px.box(
        data_frame=total_yearly_savings_fig_df,
        x="Savings",
        title="Total yearly savings")
    total_yearly_savings_fig.update_layout(xaxis_title="Total yearly savings in €", yaxis_title="")
    total_yearly_savings_fig.update_xaxes(range=[0, 10000])
    total_yearly_savings_fig.write_image(f"{images_dir}/total_yearly_savings_box.pdf")
    total_yearly_savings_fig.show()

    print("")
    print("################################")
    print("# percentual yearly savings    #")
    print("################################")
    rel_yearly_savings = 100 * rel_diff["total_yearly_costs_eur"].copy()
    rel_yearly_savings.name = "Relative yearly savings in %"
    print(rel_yearly_savings.describe().drop("count").to_markdown())
    print("-----------------------")
    n_profiles_1pct_yearly_savings = len(rel_diff[rel_diff["total_yearly_costs_eur"] > 0.01])
    print(f"Number of profiles with more than 1% yearly savings: {n_profiles_1pct_yearly_savings}")
    perc_profiles_1pct_yearly_savings = (n_profiles_1pct_yearly_savings / len(rel_diff)) * 100
    print(f"Percentage of profiles with more than 1% yearly savings: {perc_profiles_1pct_yearly_savings:.2f} %")
    n_profiles_2pct_yearly_savings = len(rel_diff[rel_diff["total_yearly_costs_eur"] > 0.02])
    print(f"Number of profiles with more than 2% yearly savings: {n_profiles_2pct_yearly_savings}")
    perc_profiles_2pct_yearly_savings = (n_profiles_2pct_yearly_savings / len(rel_diff)) * 100
    print(f"Percentage of profiles with more than 2% yearly savings: {perc_profiles_2pct_yearly_savings:.2f} %")
    n_profiles_3pct_yearly_savings = len(rel_diff[rel_diff["total_yearly_costs_eur"] > 0.03])
    print(f"Number of profiles with more than 3% yearly savings: {n_profiles_3pct_yearly_savings}")
    perc_profiles_3pct_yearly_savings = (n_profiles_3pct_yearly_savings / len(rel_diff)) * 100
    print(f"Percentage of profiles with more than 3% yearly savings: {perc_profiles_3pct_yearly_savings:.2f} %")

    perc_yearly_savings_fig_df = pd.DataFrame()
    perc_yearly_savings_fig_df = rel_diff.copy() * 100
    perc_yearly_savings_fig_df = perc_yearly_savings_fig_df.rename(columns={"total_yearly_costs_eur": "Savings"})
    perc_yearly_savings_fig = px.box(
        data_frame=perc_yearly_savings_fig_df,
        x="Savings",
        title="Relative yearly savings")
    perc_yearly_savings_fig.update_layout(xaxis_title="Relative yearly savings in %", yaxis_title="")
    perc_yearly_savings_fig.update_xaxes(range=[0, 6])
    perc_yearly_savings_fig.write_image(f"{images_dir}/perc_yearly_savings_box.pdf")
    perc_yearly_savings_fig.show()

    print("")
    print("#####################################")
    print("# total grid capacity costs savings #")
    print("#####################################")
    abs_cap_costs_yearly_savings = abs_diff["grid_capacity_costs_eur"].copy()
    abs_cap_costs_yearly_savings.name = "Total yearly grid capacity costs savings in %"
    print(abs_cap_costs_yearly_savings.describe().drop("count").to_markdown())

    total_cap_saving_fig_df = abs_diff.copy()
    total_cap_saving_fig_df = total_cap_saving_fig_df.rename(columns={"grid_capacity_costs_eur": "Savings"})
    total_cap_savings_fig = px.box(
        data_frame=total_cap_saving_fig_df,
        x="Savings",
        title="Yearly capacity costs savings")
    total_cap_savings_fig.update_layout(xaxis_title="Savings in €", yaxis_title="")
    total_cap_savings_fig.update_xaxes(range=[0, 10e3])
    total_cap_savings_fig.write_image(f"{images_dir}/total_capacity_costs_savings_box.pdf")
    total_cap_savings_fig.show()

    print("")
    print("########################################")
    print("# relative grid capacity costs savings #")
    print("########################################")
    rel_cap_costs_yearly_savings = 100 * rel_diff["grid_capacity_costs_eur"].copy()
    rel_cap_costs_yearly_savings.name = "Relative yearly grid capacity costs savings in %"
    print(rel_cap_costs_yearly_savings.describe().drop("count").to_markdown())

    rel_cap_savings_fig_df = pd.DataFrame()
    rel_cap_savings_fig_df = rel_diff.copy() * 100
    rel_cap_savings_fig_df.rename(columns={"grid_capacity_costs_eur": "Savings"}, inplace=True)
    rel_cap_savings_fig = px.box(
        data_frame=rel_cap_savings_fig_df,
        x="Savings",
        title="Relative yearly capacity cost savings")
    rel_cap_savings_fig.update_layout(xaxis_title="Savings in %", yaxis_title="")
    rel_cap_savings_fig.update_xaxes(range=[0, 50])
    rel_cap_savings_fig.write_image(f"{images_dir}/rel_capacity_costs_savings_box.pdf")
    rel_cap_savings_fig.show()

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
    fig = px.imshow(abs_correlations_df, title="Correlation coefficients for absolute yearly savings")
    fig.show()

    # show correlation coefficients
    abs_corr_fig_df = abs_correlations_df[["total_yearly_costs_eur"]].round(2)
    abs_corr_fig_df.sort_values("total_yearly_costs_eur", inplace=True, ascending=False)
    abs_corr_fig = px.bar(
        data_frame=abs_corr_fig_df,
        y="total_yearly_costs_eur",
        text_auto=True,
        title="Correlation between different load profile characteristics and absolute yearly savings")
    abs_corr_fig.update_layout(yaxis_title="Correlation coefficients", xaxis_title="Variable")
    abs_corr_fig.write_image(f"{images_dir}/abs_corr_bar.pdf")
    abs_corr_fig.show()

    # table showing absolute correlations
    df = pd.DataFrame()
    i = 0
    for var in abs_correlations_df.index:#[correlations_df["total_yearly_costs_eur"] > 0.3].index:
        corr, p_value = pearsonr(y=abs_diff_with_master.dropna()["total_yearly_costs_eur"], x=abs_diff_with_master.dropna()[var])
        df.loc[i, "var"] = var
        df.loc[i, "corr"] = corr
        df.loc[i, "p_value"] = p_value
        i += 1
    print(df.sort_values("corr", ascending=False, ignore_index=True).to_markdown())


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
    rel_corr_fig_df = rel_correlations_df[["total_yearly_costs_eur"]].round(2)
    rel_corr_fig_df.sort_values("total_yearly_costs_eur", inplace=True, ascending=False)
    rel_corr_fig_df.dropna(inplace=True)
    rel_corr_fig = px.bar(
        data_frame=rel_corr_fig_df,
        y="total_yearly_costs_eur",
        text_auto=True,
        title="Correlation between different load profile characteristics and relative yearly savings")
    rel_corr_fig.update_layout(yaxis_title="Correlation coefficient", xaxis_title="Variable")
    rel_corr_fig.write_image(f"{images_dir}/rel_corr_bar.pdf")
    rel_corr_fig.show()

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

    print(df.sort_values("corr", ascending=False, ignore_index=True).to_markdown())


if __name__ == "__main__":

    # load DB URI
    uri = os.getenv("DB_URI")

    images_dir = Path(__file__).parent / Path("images")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    compare(uri=uri, images_dir=images_dir)