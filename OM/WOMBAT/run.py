from time import perf_counter  
import numpy as np
import pandas as pd
from wombat import Simulation, load_yaml
from wombat import create_library_structure 
from time import perf_counter
from pathlib import Path
import yaml
from wombat.core import Metrics
import warnings 
warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from time import time
np.random.seed(3)
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


N_RUNS = 30
RANDOM_SEEDS = list(range(1, N_RUNS + 1))



BASE_DIR = Path("library")


def run_windfarm_simulations(config_name, random_seeds: list):
    results_dir = BASE_DIR / "results"
    if not results_dir.is_dir():
        results_dir.mkdir()

    # Initialize storage lists
    availability_records = []
    opex_records = []
    vessel_records = []
    repair_time_records = []
    power_records = []
    fixed_cost_records = []
    component_records = []

    N = len(random_seeds)
    for i, seed in enumerate(random_seeds, start=1):
        print(
            f" Running simulation {i}/N with random seed {seed}",
            end="\r",
        )

        # Run simulation
        sim = Simulation(BASE_DIR, config_name, random_seed=seed)
        sim.run(create_metrics=True, save_metrics_inputs=True)

        # Load metrics
        fpath = sim.env.metrics_input_fname.parent
        fname = sim.env.metrics_input_fname.name
        metrics = Metrics.from_simulation_outputs(fpath, fname)

        # Availability Results
        time_avail = metrics.time_based_availability(frequency="project", by="windfarm")
        prod_avail = metrics.production_based_availability(
            frequency="project", by="windfarm"
        )
        time_value = time_avail.iloc[0, 0]
        prod_value = prod_avail.iloc[0, 0]
        availability_records.append(
            {
                "run": i,
                "random_seed": seed,
                "time_based_availability": time_value,
                "production_based_availability": prod_value,
            }
        )

        # OpEx Results 
        opex_df = metrics.opex(frequency="annual", by_category=True).reset_index()
        opex_df.insert(0, "random_seed", seed)
        opex_df.insert(0, "run", i)
        opex_records.append(opex_df)

        # Vessel Costs 
        vessel_df = metrics.equipment_costs(
            frequency="annual", by_equipment=True
        ).reset_index()
        vessel_df.insert(0, "random_seed", seed)
        vessel_df.insert(0, "run", i)
        vessel_records.append(vessel_df)

        # Repair Time at Port 

        # Build full path to config file
        config_path = BASE_DIR / "project" / "config" / config_name
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        port_name = config_data.get("port", None)

        if port_name is None:
            raise KeyError(
                f"'port' key not found in {config_path}, can not calculate time at port"
            )

        port_name = port_name.replace(".yaml", "")

        events_df = sim.env.load_events_log_dataframe()
        events_df["duration"] = pd.to_numeric(events_df["duration"], errors="coerce")
        df_port = events_df[events_df["agent"] == port_name]
        total_hours = df_port["duration"].sum()
        simulation_years = sim.env.end_year - sim.env.start_year + 1
        avg_hours_per_year = total_hours / simulation_years
        avg_days_per_year = avg_hours_per_year / 24
        avg_months_per_year = avg_hours_per_year / (24 * 30.4375)

        repair_time_records.append(
            {
                "run": i,
                "random_seed": seed,
                "avg_repair_time_months": avg_months_per_year,
                "avg_repair_time_days": avg_days_per_year,
            }
        )
        
        power_df = metrics.power_production(frequency="annual", by="windfarm", units="mwh").reset_index()
        power_df.insert(0, "random_seed", seed)
        power_df.insert(0, "run", i)
        power_records.append(power_df)
        
        fixed_df = metrics.project_fixed_costs(frequency="annual", resolution="medium").reset_index()
        fixed_df.insert(0, "random_seed", seed)
        fixed_df.insert(0, "run", i)
        fixed_cost_records.append(fixed_df)
        
        component_df = metrics.component_costs(frequency="annual", by_category=True,by_action=True).reset_index()
        component_df.insert(0, "random_seed", seed)
        component_df.insert(0, "run", i)
        component_records.append(component_df)
        
        sim.env.cleanup_log_files()

    df_availability = pd.DataFrame(availability_records)
    df_opex = pd.concat(opex_records, ignore_index=True)
    df_vessels = pd.concat(vessel_records, ignore_index=True)
    df_repair_time = pd.DataFrame(repair_time_records)
    df_power = pd.concat(power_records, ignore_index=True)
    df_fixed = pd.concat(fixed_cost_records, ignore_index=True)
    df_component = pd.concat(component_records, ignore_index=True)

    config_prefix = config_name.replace(".yaml", "")

    df_availability.to_csv(
        results_dir / f"{config_prefix}_availability_results.csv",
        index=False,
    )
    df_opex.to_csv(
        results_dir / f"{config_prefix}_opex_results.csv", index=False
    )
    df_vessels.to_csv(
        results_dir / f"{config_prefix}_all_vessel_results.csv", index=False
    )
    df_repair_time.to_csv(
        results_dir / f"{config_prefix}_repair_time_at_port_results.csv",
        index=False,
    )
    df_power.to_csv(results_dir / f"{config_prefix}_power_results.csv", index=False)
    df_fixed.to_csv(results_dir / f"{config_prefix}_fixed_cost_results.csv", index=False)
    df_component.to_csv(results_dir / f"{config_prefix}_component_results.csv", index=False)
    
    print(f" All simulations complete. Results saved to {results_dir}")
    
    

configs = [f.name for f in Path("library/project/config").iterdir() if f.is_file() if f.name != "fixed_costs.yaml"]



start_time = time()
counter = 0 
n_workers = 18
print(f"Using {n_workers} processes for parallel execution")
print(f"Starting simulations at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(run_windfarm_simulations, config, RANDOM_SEEDS): config for config in configs}
    for future in tqdm(futures, desc="Running simulations for turbine configs", total=len(configs)):
        config = futures[future]
        try:
            future.result()
            counter += 1
        except Exception as e:
            print(f"Error running simulations for config {config}: {e}")

end_time = time()
elapsed_seconds = int(end_time - start_time)

print(f"\nCompleted {counter}/{len(configs)} configurations")
print(f"Total time: {elapsed_seconds} seconds")
if counter > 0:
    print(f"Average time per config: {elapsed_seconds/counter:.0f} seconds")

