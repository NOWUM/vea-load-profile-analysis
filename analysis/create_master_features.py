import os
import logging

import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


log = logging.getLogger(__name__)

def create_features(uri: str):

    # load master table
    master = pd.read_sql("SELECT * FROM vea_industrial_load_profiles.master", uri)

    # create sql query
    sql = f"SELECT id, value FROM vea_industrial_load_profiles.load WHERE id = %s"

    # iterate over profile IDs and create features for each profile
    for id in tqdm(master["id"].values):
            
            try:
                load_ts = pd.read_sql(sql, params=(int(id),), con=uri, index_col="id")
                master.loc[master["id"] == id, "peak_load_kw"] = load_ts["value"].max()
                master.loc[master["id"] == id, "mean_load_kw"] = load_ts["value"].mean()
                master.loc[master["id"] == id, "variance_kw"] = load_ts["value"].var()
                master.loc[master["id"] == id, "total_energy_kwh"] = load_ts["value"].sum() / 4
                master.loc[master["id"] == id, "full_load_hours_h"] = (load_ts["value"].sum() / 4) / load_ts["value"].max()
                master.loc[master["id"] == id, "is_over_2500h"] = ((load_ts["value"].sum() / 4) / load_ts["value"].max()) > 2500
                master.loc[master["id"] == id, "std_kw"] = load_ts["value"].std()
            except Exception as e:
                 msg = f"Error on ID {id}: {e}"
                 log.error(msg)

    # try to save the dataframe
    try:
        master.to_sql(
            name="master_with_features",
            con=uri,
            schema="vea_industrial_load_profiles",
            if_exists="fail")
    except Exception as e:
        # log error and save to .csv if it fails
        msg = f"Could not write dataframe to SQL: {e}.\n" 
        msg += f"Writing to 'master_with_features.csv' instead."
        log.error(msg)
        master.to_csv('master_with_features.csv')


if __name__ == "__main__":

    # load DB URI
    uri = os.getenv("DB_URI")
    
    create_features(uri)