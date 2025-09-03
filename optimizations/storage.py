import os
import logging

import pandas as pd
from peakshaving_analyzer import load_oeds_config, PeakShavingAnalyzer

from dotenv import load_dotenv
load_dotenv()

def calculate_baselines():
    URI = os.getenv("DB_URI")

    max_id = pd.read_sql("SELECT id FROM vea_industrial_load_profiles.master", URI)["id"].max()
    profile_ids = list(range(max_id + 1))

    for profile_id in profile_ids:
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

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    calculate_baselines()