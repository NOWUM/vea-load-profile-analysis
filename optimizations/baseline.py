import os
import logging
import sys

from multiprocessing import Pool

import pandas as pd
from peakshaving_analyzer import load_oeds_config, PeakShavingAnalyzer

from dotenv import load_dotenv
load_dotenv()

URI = os.getenv("DB_URI")

log = logging.getLogger(__name__)


def calculate_profile(profile_id: int):
    try:
        log.info(f"Calculating baseline for {profile_id=}")
        config = load_oeds_config(
            con=URI,
            profile_id=profile_id,
            name=f"{profile_id}_baseline",
            producer_energy_price=0.18,
            storage_cost_per_kwh=285,
            interest_rate=2,
            storage_charge_efficiency=0.95,
            storage_discharge_efficiency=0.95,
            storage_charge_rate=5,
            storage_discharge_rate=5,
            add_storage=False,
            add_solar=False,
            solver="gurobi",
            verbose=False)

        df = pd.DataFrame()
        df["name"] = [str(profile_id) + "_baseline"]
        df["energy_cost_eur"] = [config.producer_energy_price * config.consumption_timeseries.sum() * config.hours_per_timestep]
        df["grid_energy_cost_eur"] = [config.grid_energy_price * config.consumption_timeseries.sum() * config.hours_per_timestep]
        df["grid_capacity_cost_eur"] = [config.grid_capacity_price * config.consumption_timeseries.max()]
        df["grid_capacity_kw"] = [config.consumption_timeseries.max()]
        df["storage_invest_eur"] = [0]
        df["storage_annuity_eur"] = [0]
        df["storage_capacity_kwh"] = [0]
        df["inverter_invest_eur"] = [0]
        df["inverter_annuity_eur"] = [0]
        df["inverter_capacity_kw"] = [0]
        df["solar_invest_eur"] = [0]
        df["solar_annuity_eur"] = [0]
        df["solar_capacity_kwp"] = [0]
        df["total_yearly_costs_eur"] = df["energy_cost_eur"] + df["grid_energy_cost_eur"] + df["grid_capacity_cost_eur"]
        df["total_annuity_eur"] = [0]
        df["total_invest_eur"] = [0]

        return df

        # df.to_sql(name="overview", con=URI, schema="vea_results", if_exists="append", index=False)

    except Exception as e:
        raise e

def calculate_baselines(n_processes):

    max_id = pd.read_sql("SELECT id FROM vea_industrial_load_profiles.master", URI)["id"].max()
    profile_ids = list(range(max_id + 1))

    with Pool(processes=n_processes) as pool:
        pool.map(calculate_profile, profile_ids)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.ERROR,
        datefmt='%Y-%m-%d %H:%M:%S')

    if len(sys.argv) > 1:
        n_processes = int(sys.argv[1])
    else:
         n_processes=2

    log.info(f"Starting calculation of all profiles with {n_processes=}")
    calculate_baselines(n_processes)