import os
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import pearsonr

from dotenv import load_dotenv
load_dotenv()

def compare(uri: str, images_dir: str | Path):

    # read in masters data
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

    # read in baseline data
    baseline = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%base%%'", uri)
    baseline["id"] = baseline["name"].str.split("_").str[0].astype(int)
    baseline.set_index("id", inplace=True)
    baseline.sort_index(inplace=True)
    print(baseline.head())

    # read in storage_pv data
    storage_pv = pd.read_sql("SELECT * FROM vea_results.overview WHERE name LIKE '%%storage_pv'", uri)
    storage_pv["id"] = storage_pv["name"].str.split("_").str[0].astype(int)
    storage_pv.set_index("id", inplace=True)
    storage_pv.sort_index(inplace=True)
    print(storage_pv.head())

    print("")
    print("################################")
    print("#       general analysis       #")
    print("################################")
    total_profiles_analyzed = len(storage_pv)
    profiles_using_pv = storage_pv[storage_pv["inverter_invest_eur"] > 0]
    n_profiles_using_pv = len(profiles_using_pv)

    print(f"{total_profiles_analyzed=}")
    print(f"{n_profiles_using_pv=}")
    print(f"Percentage of profiles using PV: {((n_profiles_using_pv / total_profiles_analyzed) * 100):.2f} %")

    print("")
    print("################################")
    print("#        system sizes          #")
    print("################################")
    stor_cap_col = "storage_capacity_kwh"
    median_storage_size = profiles_using_pv[stor_cap_col].median()
    print(f"{median_storage_size=:.4f} kWh")
    mean_storage_size = profiles_using_pv[stor_cap_col].mean()
    print(f"{mean_storage_size=:.4f} kWh")
    min_storage_size = profiles_using_pv[stor_cap_col].min()
    print(f"{min_storage_size=:.4f} kWh")
    max_storage_size = profiles_using_pv[stor_cap_col].max()
    print(f"{max_storage_size=:.4f} kWh")
    q95_storage_size = profiles_using_pv[stor_cap_col].quantile(0.95)
    print(f"{q95_storage_size=:.2f} kWh")

    print("----------------------------------")
    inv_cap_col = "inverter_capacity_kw"
    median_inverter_size = profiles_using_pv[inv_cap_col].median()
    print(f"{median_inverter_size=:.2f} kW")
    mean_inverter_size = profiles_using_pv[inv_cap_col].mean()
    print(f"{mean_inverter_size=:.2f} kW")
    min_inverter_size = profiles_using_pv[inv_cap_col].min()
    print(f"{min_inverter_size=:.2f} kW")
    max_inverter_size = profiles_using_pv[inv_cap_col].max()
    print(f"{max_inverter_size=:.2f} kW")
    q95_inverter_size = profiles_using_pv[inv_cap_col].quantile(0.95)
    print(f"{q95_inverter_size=:.2f} kW")

    print("----------------------------------")
    pv_cap_col = "solar_capacity_kwp"
    median_solar_size = profiles_using_pv[pv_cap_col].median()
    print(f"{median_solar_size=:.2f} kWp")
    mean_solar_size = profiles_using_pv[pv_cap_col].mean()
    print(f"{mean_solar_size=:.2f} kWp")
    min_solar_size = profiles_using_pv[pv_cap_col].min()
    print(f"{min_solar_size=:.2f} kWp")
    max_solar_size = profiles_using_pv[pv_cap_col].max()
    print(f"{max_solar_size=:.2f} kWp")

    system_size_fig_df = profiles_using_pv.copy()
    system_size_fig_df = system_size_fig_df.rename(columns={
        stor_cap_col: "Storage",
        inv_cap_col: "Inverter",
        pv_cap_col: "PV system"})
    system_size_fig = px.box(
        data_frame=system_size_fig_df,
        x=["Inverter", "Storage", "PV system"],
        title="System sizes")
    system_size_fig.update_layout(xaxis_title="Capacity in kWh (storage) / kW (inverter, PV)", yaxis_title="")
    system_size_fig.update_xaxes(range=[0, 15e3])
    system_size_fig.write_image(f"{images_dir}/system_sizes.pdf")
    system_size_fig.show()

    print("")
    print("################################")
    print("#    system investments        #")
    print("################################")
    stor_inv_col = "storage_invest_eur"
    median_storage_invest = profiles_using_pv[stor_inv_col].median()
    print(f"{median_storage_invest=:.2f} €")
    mean_storage_invest = profiles_using_pv[stor_inv_col].mean()
    print(f"{mean_storage_invest=:.2f} €")
    min_storage_invest = profiles_using_pv[stor_inv_col].min()
    print(f"{min_storage_invest=:.2f} €")
    max_storage_invest = profiles_using_pv[stor_inv_col].max()
    print(f"{max_storage_invest=:.2f} €")
    q95_storage_invest = profiles_using_pv[stor_inv_col].quantile(0.95)
    print(f"{q95_storage_invest=:.2f} €")

    print("----------------------------------")
    inv_invest_col = "inverter_invest_eur"
    median_inverter_invest = profiles_using_pv[inv_invest_col].median()
    print(f"{median_inverter_invest=:.2f} €")
    mean_inverter_invest = profiles_using_pv[inv_invest_col].mean()
    print(f"{mean_inverter_invest=:.2f} €")
    min_inverter_invest = profiles_using_pv[inv_invest_col].min()
    print(f"{min_inverter_invest=:.2f} €")
    max_inverter_invest = profiles_using_pv[inv_invest_col].max()
    print(f"{max_inverter_invest=:.2f} €")
    q95_inverter_invest = profiles_using_pv[inv_invest_col].quantile(0.95)
    print(f"{q95_inverter_invest=:.2f} €")

    print("----------------------------------")
    pv_invest_col = "solar_invest_eur"
    median_solar_invest = profiles_using_pv[pv_invest_col].median()
    print(f"{median_solar_invest=:.2f} €")
    mean_solar_invest = profiles_using_pv[pv_invest_col].mean()
    print(f"{mean_solar_invest=:.2f} €")
    min_solar_invest = profiles_using_pv[pv_invest_col].min()
    print(f"{min_solar_invest=:.2f} €")
    max_solar_invest = profiles_using_pv[pv_invest_col].max()
    print(f"{max_solar_invest=:.2f} €")

    system_invest_fig_df = profiles_using_pv.copy()
    system_invest_fig_df = system_invest_fig_df.rename(columns={
        stor_inv_col: "Storage",
        inv_invest_col: "Inverter",
        pv_invest_col: "PV system"})
    system_invest_fig = px.box(
        data_frame=system_invest_fig_df,
        x=["Inverter", "Storage", "PV system"],
        title="System investments")
    system_invest_fig.update_layout(xaxis_title="System investments in €", yaxis_title="")
    system_invest_fig.update_xaxes(range=[0, 7e6])
    system_invest_fig.write_image(f"{images_dir}/system_invest_box.pdf")
    system_invest_fig.show()

    print("")
    print("################################")
    print("#     total yearly savings     #")
    print("################################")
    abs_diff = baseline.drop(columns="name") - storage_pv.drop(columns="name")
    print(abs_diff.head())

    rel_diff = (baseline.drop(columns="name") - storage_pv.drop(columns="name")) / baseline.drop(columns="name")
    # drop those that could not be optimized
    rel_diff.dropna(subset="total_yearly_costs_eur", inplace=True)
    print(rel_diff.head())

    tot_y_sav_col = "total_yearly_costs_eur"
    median_yearly_savings = abs_diff[tot_y_sav_col].median()
    print(f"{median_yearly_savings=:.2f} €")
    mean_yearly_savings = abs_diff[tot_y_sav_col].mean()
    print(f"{mean_yearly_savings=:.2f} €")
    min_yearly_savings = abs_diff[tot_y_sav_col].min()
    print(f"{min_yearly_savings=:.2f} €")
    max_yearly_savings = abs_diff[tot_y_sav_col].max()
    print(f"{max_yearly_savings=:.2f} €")
    q95_yearly_savings = abs_diff[tot_y_sav_col].quantile(0.95)
    print(f"{q95_yearly_savings=:.2f} €")

    abs_tot_yearly_savings_fig_df = abs_diff.copy()
    abs_tot_yearly_savings_fig_df = abs_tot_yearly_savings_fig_df.rename(columns={tot_y_sav_col: "Savings"})
    abs_tot_yearly_savings_fig = px.box(
        data_frame=abs_tot_yearly_savings_fig_df,
        x="Savings",
        title="Total yearly savings")
    abs_tot_yearly_savings_fig.update_layout(xaxis_title="Total yearly savings in €", yaxis_title="")
    abs_tot_yearly_savings_fig.update_xaxes(range=[0, 500e3])
    abs_tot_yearly_savings_fig.write_image(f"{images_dir}/abs_yearly_savings_box.pdf")
    abs_tot_yearly_savings_fig.show()

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

    rel_yearly_savings_fig_df = pd.DataFrame()
    rel_yearly_savings_fig_df["Savings"] = perc_yearly_savings.copy()
    rel_yearly_savings_fig_df = rel_yearly_savings_fig_df.rename(columns={"Savings": "Savings"})
    rel_yearly_savings_fig = px.box(
        data_frame=rel_yearly_savings_fig_df,
        x="Savings",
        title="Relative yearly savings")
    rel_yearly_savings_fig.update_layout(xaxis_title="Relative yearly savings in %", yaxis_title="")
    rel_yearly_savings_fig.update_xaxes(range=[10, 50])
    rel_yearly_savings_fig.show()

    print("")
    print("#################################")
    print("# absolute correlation analysis #")
    print("#################################")
    # merge savings onto master (with features)
    abs_diff_with_master = pd.merge(left=abs_diff, right=master, how="left", left_index=True, right_index=True)
    abs_diff_with_master.head()

    abs_diff_with_master["std_by_mean"] = abs_diff_with_master["std_kw"] / abs_diff_with_master["mean_load_kw"]
    abs_diff_with_master["std_by_peak"] = abs_diff_with_master["std_kw"] / abs_diff_with_master["peak_load_kw"]
    abs_diff_with_master["peak_by_mean"] = abs_diff_with_master["peak_load_kw"] / abs_diff_with_master["mean_load_kw"]

    cols_to_drop = [
        "grid_level",
        "zip_code",
        "sector_group_id",
        "sector_group",
        "solar_invest_eur",
        "solar_annuity_eur",
        "solar_capacity_kwp"]
    abs_correlations_df = abs_diff_with_master.drop(columns=cols_to_drop).corr()
    px.imshow(abs_correlations_df, title="Correlation coefficients for total yearly savings")

    abs_corr_fig_df = abs_correlations_df[["total_yearly_costs_eur"]].round(2)
    abs_corr_fig_df.sort_values("total_yearly_costs_eur", inplace=True, ascending=False)
    abs_corr_fig = px.bar(
        data_frame=abs_corr_fig_df,
        y="total_yearly_costs_eur",
        text_auto=True,
        title="Correlation between different load profile characteristics and total yearly savings")
    abs_corr_fig.update_layout(yaxis_title="Correlation coefficient", xaxis_title="Variable")
    abs_corr_fig.write_image(f"{images_dir}/abs_corr_bar.pdf")
    abs_corr_fig.show()

    df = pd.DataFrame()
    i = 0
    for var in abs_correlations_df.index:
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

    rel_diff_with_master["std_by_mean"] = rel_diff_with_master["std_kw"] / rel_diff_with_master["mean_load_kw"]
    rel_diff_with_master["std_by_peak"] = rel_diff_with_master["std_kw"] / rel_diff_with_master["peak_load_kw"]
    rel_diff_with_master["peak_by_mean"] = rel_diff_with_master["peak_load_kw"] / rel_diff_with_master["mean_load_kw"]

    cols_to_drop = [
        "grid_level",
        "zip_code",
        "sector_group_id",
        "sector_group",
        "solar_invest_eur",
        "solar_annuity_eur",
        "solar_capacity_kwp"]
    rel_correlations_df = rel_diff_with_master.drop(columns=cols_to_drop).corr()
    px.imshow(rel_correlations_df, title="Correlation coefficients for relative yearly savings")

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

    # load DB uri
    uri = os.getenv("DB_URI")

    images_dir = Path(__file__).parent / Path("images")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    compare(uri=uri, images_dir=images_dir)
