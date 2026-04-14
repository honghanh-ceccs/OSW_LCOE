from time import perf_counter  
import numpy as np
import pandas as pd
from wombat import Simulation, load_yaml
from wombat import create_library_structure 
from time import perf_counter
from pathlib import Path
import yaml
from wombat.core import Metrics
from wombat.core import Failure
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


N_RUNS = 2
RANDOM_SEEDS = list(range(1, N_RUNS + 1))

# This file is for testing a single turbine config with typhoon events

BASE_DIR = Path("library")

def load_storm_requests_for_seed(
    base_dir: Path,
    seed: int,
    config_name: str | None = None,
) -> pd.DataFrame:
    """Load storm requests with config-specific priority and sensible fallbacks."""
    storm_dir = base_dir / "project" / "storm_events"
    config_prefix = ""
    if config_name:
        config_prefix = config_name.replace(".yaml", "")

    candidates = []
    if config_prefix:
        candidates.append(storm_dir / f"storm_failure_requests_{config_prefix}_seed_{seed}.csv")
    candidates.append(storm_dir / f"storm_failure_requests_seed_{seed}.csv")
    if config_prefix:
        candidates.append(storm_dir / f"storm_failure_requests_{config_prefix}.csv")
    candidates.append(storm_dir / "storm_failure_requests.csv")

    f = next((path for path in candidates if path.exists()), None)
    if f is None:
        return pd.DataFrame()

    df = pd.read_csv(f)
    if "event_frequency" not in df.columns:
        return pd.DataFrame()
    print(f"[seed {seed}] Loaded storm file: {f.name}")
    return df.reset_index(drop=True)


def _resolve_subassembly(system, component_name: str):
    """Find a subassembly by id/name (case-insensitive, underscores normalized)."""
    target = str(component_name).strip().lower().replace(" ", "_")
    for sub in system.subassemblies:
        sub_id = str(sub.id).strip().lower().replace(" ", "_")
        sub_name = str(sub.name).strip().lower().replace(" ", "_")
        if target in (sub_id, sub_name):
            return sub
    return None


def _to_event_hour(value, sim) -> float | None:
    """Convert event_frequency to simulation hour offset.

    Supported formats:
    - datetime string/timestamp: interpreted against simulation start datetime
    - numeric <= simulation max hours: interpreted as hour offset
    - numeric in [0, simulation_years + 1]: interpreted as years from start
    """
    if pd.isna(value):
        return None

    dt = pd.to_datetime(value, errors="coerce")
    if pd.notna(dt):
        start = pd.Timestamp(sim.env.start_datetime)
        end = pd.Timestamp(sim.env.end_datetime)
        if start <= dt <= end:
            return (dt - start).total_seconds() / 3600.0

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None

    numeric = float(numeric)
    if 0 <= numeric <= float(sim.env.max_run_time):
        return numeric
    if 0 <= numeric <= float(getattr(sim.env, "simulation_years", 0) + 1):
        return numeric * 8760.0
    return None


def _inject_storm_failures(sim: Simulation, storm_df: pd.DataFrame, seed: int) -> None:
    """Inject storm failures as unscheduled repair requests during the simulation."""
    if storm_df.empty:
        return

    required = {
        "event_frequency",
        "turbine_id",
        "component",
        "failure_mode",
        "repair_time_hr",
        "materials_usd",
        "service_equipment",
        "operation_reduction",
    }
    missing = required.difference(storm_df.columns)
    if missing:
        print(f"[seed {seed}] Missing storm columns: {sorted(missing)}. Skipping storm injection.")
        return

    # Fail fast if IDs are incompatible with the active layout.
    layout_turbine_ids = set(map(str, sim.windfarm.turbine_id))
    storm_turbine_ids = set(storm_df["turbine_id"].astype(str).str.strip())
    if not storm_turbine_ids.intersection(layout_turbine_ids):
        print(
            f"[seed {seed}] Injected 0 storm failures "
            f"(ID mismatch: no overlap between storm and layout turbine IDs)."
        )
        return

    rows = []
    for row in storm_df.itertuples(index=False):
        event_hour = _to_event_hour(getattr(row, "event_frequency", None), sim)
        if event_hour is None:
            continue
        rows.append((event_hour, row))

    if not rows:
        print(f"[seed {seed}] No valid storm events in file.")
        return

    rows.sort(key=lambda x: x[0])

    def _storm_event_process():
        skipped_system = 0
        skipped_component = 0
        injected = 0

        for event_hour, row in rows:
            if event_hour > sim.env.max_run_time:
                continue

            wait_hours = max(0.0, event_hour - sim.env.now)
            if wait_hours > 0:
                yield sim.env.timeout(wait_hours)

            system_id = str(getattr(row, "turbine_id", "")).strip()
            try:
                system = sim.windfarm.system(system_id)
            except Exception:
                skipped_system += 1
                continue

            subassembly = _resolve_subassembly(system, getattr(row, "component", ""))
            if subassembly is None:
                skipped_component += 1
                continue

            service_equipment = str(getattr(row, "service_equipment", "CTV"))
            service_equipment = [
                item.strip() for item in service_equipment.split("|") if item.strip()
            ] or ["CTV"]

            operation_reduction = float(
                pd.to_numeric(getattr(row, "operation_reduction", 1.0), errors="coerce")
            )
            operation_reduction = min(max(operation_reduction, 0.0), 1.0)

            replacement = bool(operation_reduction >= 0.999)
            severity = 5 if replacement else 3

            action = Failure(
                scale=1.0,
                shape=1.0,
                time=float(pd.to_numeric(getattr(row, "repair_time_hr", 0), errors="coerce") or 0.0),
                materials=float(pd.to_numeric(getattr(row, "materials_usd", 0), errors="coerce") or 0.0),
                operation_reduction=operation_reduction,
                level=severity,
                service_equipment=service_equipment,
                system_value=system.value,
                replacement=replacement,
                description=str(getattr(row, "failure_mode", "storm failure")),
                rng=sim.env.random_generator,
            )
            subassembly.trigger_request(action)
            injected += 1

        print(
            f"[seed {seed}] Injected {injected} storm failures "
            f"(skipped system={skipped_system}, component={skipped_component})."
        )

    sim.env.process(_storm_event_process())

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
        storm_df = load_storm_requests_for_seed(BASE_DIR, seed, config_name=config_name)
        _inject_storm_failures(sim, storm_df, seed)
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
    
    

#configs = [f.name for f in Path("library/project/config").iterdir() if f.is_file() if f.name != "fixed_costs.yaml"]

configs = ["IEA 15 MW Reference_base.yaml"] 

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

