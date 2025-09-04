import os
import logging
import sys

from multiprocessing import Pool

import pandas as pd
from peakshaving_analyzer import load_oeds_config, PeakShavingAnalyzer

from dotenv import load_dotenv
load_dotenv()

URI = os.getenv("DB_URI")

def optimize_profile(profile_id: int):
        try:
            log.info(f"Calculating baseline for {profile_id=}")
            config = load_oeds_config(
                con=URI,
                profile_id=profile_id,
                name=f"{profile_id}_storage",
                add_storage=True,
                solver="gurobi",
                verbose=True)
            psa = PeakShavingAnalyzer(config=config)
            results = psa.optimize()
            results.to_sql(connection=URI, schema="vea_results")
            
        except Exception as e:
            log.error(e)

def calculate_baselines(n_processes):

    max_id = pd.read_sql("SELECT id FROM vea_industrial_load_profiles.master", URI)["id"].max()
    profile_ids = list(range(max_id + 1))

    with Pool(processes=n_processes) as pool:
        pool.map(optimize_profile, profile_ids)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

    if len(sys.argv) > 1:
        n_processes = int(sys.argv[1])
    else:
         n_processes=2

    log.info(f"Starting calculation of all profiles with {n_processes=}")
    calculate_baselines(n_processes)