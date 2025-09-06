## What is this?
Code in this repository analyzes the [VEA Industrial Load Profiles](https://zenodo.org/records/13910298) dataset regarding peak-shaving utilizing the [PeakShavingAnalyzer](https://github.com/NOWUM/peakshaving-analyzer).

It queries the necessary data from [OpenEnergyDataServer](https://github.com/open-energy-data-server/open-energy-data-server) and analyzes it regarding system sizes, investments and savings. Furthermore, a basic correlation analysis has been carried out.

In the directory `optimization`, the load profiles are optimized. Each scenario (baseline, storage, storage + PV) has it's own optimization file.

In the directory `analysis`, the scenarios `storage` and `storage_pv` are compared to the scenario `baseline`.

## How to analyse yourself
1. Create a `.env` file stating the `DB_URI` for your OEDS.
2. Create the metadata needed by running `analysis/create_master_features.py`
3. Optimize the profiles using the files in `/optimizations`
4. Analyse the profiles using the files in `/analysis`