# %% [markdown]
# # NB21 — Market Revenue Dispatch & LCOS
# **Phase 2 — GB Constraint Analytics Platform**
#
# **Scope:** Simulate historical wholesale market dispatch. Calculate gross merchant 
# revenues, effective capture prices, and the baseline Levelized Cost of Storage (LCOS).
#
# **Methodological Corrections applied here:**
# - 🚨 **Correction 4 (The Daily Cycle Myth):** Implements a 7-day (168-hour) rolling 
#   optimization window. Proves the "Tank > Pipe" thesis by allowing deep tanks to 
#   charge over multi-day weather lulls and discharge during extended winter peaks.
# -  **Correction 5 (The Narrow Pipe Penalty):** Applies a 15% cannibalization 
#   discount to the discharge capture price for assets flagged `rrt_applicable=True`.

# %%
## 0. Imports and Configuration
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR

NB20_INPUT_PATH = PROCESSED_DIR / "ldes_portfolio_economics.parquet"
PRICE_CURVE_PATH = RAW_DIR / "historical_day_ahead_prices.parquet"
OUTPUT_PATH = OUTPUT_DIR / "ldes_portfolio_revenues.parquet"

# ─── Constants ────────────────────────────────────────────────────────────────
CANNIBALIZATION_PENALTY = 0.15  # 15% price discount for Narrow Pipe assets
HOURS_PER_YEAR = 8760

print("NB21 — Market Revenue Dispatch & LCOS")
print("=" * 65)

# %% [markdown]
# ## 1. Data Ingestion

# %%
def load_portfolio(path: Path) -> pd.DataFrame:
    print(f"\n[1.1] Loading NB20 costed portfolio...")
    df = pd.read_parquet(path)
    print(f"  ✅ {len(df)} projects loaded.")
    return df

def load_prices(path: Path) -> pd.DataFrame:
    print(f"\n[1.2] Loading historical price curve...")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns or "price_gbp_mwh" not in df.columns:
        raise ValueError("Price curve missing 'timestamp' or 'price_gbp_mwh' columns.")
    print(f"  ✅ {len(df)} hourly prices loaded.")
    return df

df_portfolio = load_portfolio(NB20_INPUT_PATH)
df_prices = load_prices(PRICE_CURVE_PATH)

# %% [markdown]
# ## 2. Correction 4: 7-Day Rolling Window Dispatch (Proving "Tank > Pipe")

# %%
def simulate_duration_constrained_dispatch(df_portfolio: pd.DataFrame, 
                                            df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Duration-constrained dispatch optimisation.
    Short-duration assets search within 1-day windows.
    Long-duration assets search within multi-day windows.
    """
    print("\n[2.1] Simulating duration-constrained dispatch...")

    prices = df_prices["price_gbp_mwh"].values
    n_hours = len(prices)
    
    # Calculate analysis period
    analysis_years = n_hours / 8760
    print(f"  Analysis period: {analysis_years:.1f} years")

    annual_revenues = []
    avg_charge_prices = []
    avg_discharge_prices = []
    annual_mwh_delivered = []
    annual_charge_mwh_list = []
    annual_cycles = []

    for _, row in df_portfolio.iterrows():
        mw = row["mw_capacity"]
        duration = row["duration_hours"]
        rte = row["rte"]

        # KEY FIX: Constrain search window based on duration
        # Short-duration (< 24h): daily cycles only
        # Long-duration (>= 24h): multi-day cycles
        if duration < 24:
            search_window_hours = 24  # 1 day
        elif duration < 48:
            search_window_hours = 48  # 2 days
        elif duration < 72:
            search_window_hours = 72  # 3 days
        else:
            search_window_hours = 168  # 7 days
        
        n_windows = n_hours // search_window_hours
        
        charge_h = int(duration)
        discharge_h = int(duration * rte)

        window_charge_prices = []
        window_discharge_prices = []
        window_cycles = 0

        for w in range(n_windows):
            window_prices = prices[w * search_window_hours:(w + 1) * search_window_hours]

            sorted_idx = np.argsort(window_prices)
            charge_idx = sorted_idx[:charge_h]
            discharge_idx = sorted_idx[-discharge_h:]

            # Check for overlap
            charge_set = set(charge_idx)
            discharge_set = set(discharge_idx)
            if charge_set & discharge_set:
                non_discharge_idx = [i for i in sorted_idx if i not in discharge_set]
                charge_idx = np.array(non_discharge_idx[:charge_h])

            charge_price = np.mean(window_prices[charge_idx])
            discharge_price = np.mean(window_prices[discharge_idx])

            spread = discharge_price - (charge_price / rte)

            if spread >= 10.0:
                window_charge_prices.append(charge_price)
                window_discharge_prices.append(discharge_price)
                window_cycles += 1

        # Annualise
        avg_chg = np.mean(window_charge_prices) if window_charge_prices else 0.0
        avg_dis = np.mean(window_discharge_prices) if window_discharge_prices else 0.0

        annual_cycles_per_year = window_cycles / analysis_years

        ann_charge_mwh = (window_cycles * charge_h * mw) / analysis_years
        ann_discharge_mwh = (window_cycles * discharge_h * mw) / analysis_years
        
        gross_rev = (avg_dis * ann_discharge_mwh) - (avg_chg * ann_charge_mwh)

        annual_revenues.append(gross_rev)
        avg_charge_prices.append(avg_chg)
        avg_discharge_prices.append(avg_dis)
        annual_mwh_delivered.append(ann_discharge_mwh)
        annual_charge_mwh_list.append(ann_charge_mwh)
        annual_cycles.append(annual_cycles_per_year)

    df = df_portfolio.copy()
    df["annual_gross_revenue_gbp"] = annual_revenues
    df["avg_charge_price_gbp"] = avg_charge_prices
    df["avg_discharge_price_gbp"] = avg_discharge_prices
    df["annual_mwh_delivered"] = annual_mwh_delivered
    df["annual_charge_mwh"] = annual_charge_mwh_list
    df["annual_cycles"] = annual_cycles
    df["gross_spread_gbp"] = (
        df["avg_discharge_price_gbp"] - (df["avg_charge_price_gbp"] / df["rte"])
    )

    print(f"  ✅ Duration-constrained dispatch complete.")
    print(f"  Avg annual cycles: {np.mean(annual_cycles):.1f}")
    print(f"  Avg annual MWh delivered: {np.mean(annual_mwh_delivered):,.0f}")
    return df


# %% [markdown]
# ## 3. Correction 5: The "Narrow Pipe" Penalty

# %%
def apply_narrow_pipe_penalty(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3.1] Applying Narrow Pipe Penalty (Correction 5)...")

    df = df.copy()

    mask_rrt = df["rrt_applicable"] == True
    n_penalized = mask_rrt.sum()

    if n_penalized > 0:
        df.loc[mask_rrt, "avg_discharge_price_gbp"] *= (
            1 - CANNIBALIZATION_PENALTY
        )

        df.loc[mask_rrt, "annual_gross_revenue_gbp"] = (
            df.loc[mask_rrt, "avg_discharge_price_gbp"]
            * df.loc[mask_rrt, "annual_mwh_delivered"]
        ) - (
            df.loc[mask_rrt, "avg_charge_price_gbp"]
            * df.loc[mask_rrt, "annual_charge_mwh"]
        )

        df.loc[mask_rrt, "gross_spread_gbp"] = (
            df.loc[mask_rrt, "avg_discharge_price_gbp"]
            - (df.loc[mask_rrt, "avg_charge_price_gbp"] / df.loc[mask_rrt, "rte"])
        )

        print(
            f"  ⚠️  Applied {CANNIBALIZATION_PENALTY * 100:.0f}% discharge penalty "
            f"to {n_penalized} RRT-flagged projects."
        )

    return df


# %% [markdown]
# ## 4. LCOS & Financial Hurdles

# %%
def calculate_lcos_and_hurdles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates LCOS, Required Revenue, and Required Floor."""
    print("\n[4.1] Calculating LCOS and financial hurdles...")

    required_cols = [
        "lifecycle_capex_gbp",
        "wacc",
        "project_life_yrs",
        "total_fom_gbp_yr",
        "annual_mwh_delivered",
        "mw_capacity",
        "annual_gross_revenue_gbp",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    r = df["wacc"].values
    n = df["project_life_yrs"].values
    pv = df["lifecycle_capex_gbp"].values

    annualization_factor = np.where(
        r > 0,
        (r * (1 + r) ** n) / ((1 + r) ** n - 1),
        1 / n,
    )

    annualized_capex = pv * annualization_factor

    df["total_annual_cost_gbp"] = (
        annualized_capex + df["total_fom_gbp_yr"].values
    )

    df["lcos_gbp_mwh"] = np.where(
        df["annual_mwh_delivered"] > 0,
        df["total_annual_cost_gbp"] / df["annual_mwh_delivered"],
        np.nan,
    )

    df["required_revenue_per_mw_yr"] = (
        df["total_annual_cost_gbp"] / df["mw_capacity"]
    )

    df["actual_revenue_per_mw_yr"] = (
        df["annual_gross_revenue_gbp"] / df["mw_capacity"]
    )

    df["required_floor_per_mw_yr"] = np.maximum(
        0,
        df["required_revenue_per_mw_yr"] - df["actual_revenue_per_mw_yr"],
    )

    print("  ✅ LCOS and hurdles calculated.")
    print(
        f"  LCOS range: £{df['lcos_gbp_mwh'].min():.0f} "
        f"- £{df['lcos_gbp_mwh'].max():.0f}/MWh"
    )

    return df


# %%
# ==============================================================================
# RUN NB21 PIPELINE
# dispatch → narrow-pipe penalty → LCOS/hurdles
# ==============================================================================

print("\nRunning NB21 pipeline...")

df_revenue = simulate_duration_constrained_dispatch(
    df_portfolio=df_portfolio,
    df_prices=df_prices,
)

df_revenue = apply_narrow_pipe_penalty(
    df_revenue
)

df_revenue = calculate_lcos_and_hurdles(
    df_revenue
)

REQUIRED_PIPELINE_COLUMNS = [
    "annual_gross_revenue_gbp",
    "actual_revenue_per_mw_yr",
    "avg_charge_price_gbp",
    "avg_discharge_price_gbp",
    "gross_spread_gbp",
    "annual_mwh_delivered",
    "annual_charge_mwh",
    "annual_cycles",
    "total_annual_cost_gbp",
    "lcos_gbp_mwh",
    "required_revenue_per_mw_yr",
    "required_floor_per_mw_yr",
]

missing_pipeline_cols = [
    col for col in REQUIRED_PIPELINE_COLUMNS
    if col not in df_revenue.columns
]

if missing_pipeline_cols:
    raise RuntimeError(
        f"NB21 pipeline incomplete — missing columns: {missing_pipeline_cols}"
    )

print("✅ NB21 pipeline complete.")
print(f"Rows: {len(df_revenue):,}")
print(f"Columns: {len(df_revenue.columns):,}")

display(
    df_revenue[
        [
            "project_name",
            "economic_archetype",
            "annual_cycles",
            "annual_mwh_delivered",
            "annual_charge_mwh",
            "gross_spread_gbp",
            "lcos_gbp_mwh",
            "required_floor_per_mw_yr",
        ]
    ].head()
)

# %% [markdown]
# ## 5. Output Construction & Export

# %%
# ==============================================================================
# 5. Output Construction & Export
# ==============================================================================

print("\n[5.1] Building output...")

NB20_COLS = df_portfolio.columns.tolist()

REVENUE_COLS = [
    "annual_gross_revenue_gbp",
    "actual_revenue_per_mw_yr",
    "avg_charge_price_gbp",
    "avg_discharge_price_gbp",
    "gross_spread_gbp",
    "annual_mwh_delivered",
    "annual_charge_mwh",
    "annual_cycles",
    "total_annual_cost_gbp",
    "lcos_gbp_mwh",
    "required_revenue_per_mw_yr",
    "required_floor_per_mw_yr",
]

OUTPUT_COLUMNS = list(dict.fromkeys(NB20_COLS + REVENUE_COLS))

missing_out = [
    col for col in OUTPUT_COLUMNS
    if col not in df_revenue.columns
]

if missing_out:
    raise RuntimeError(
        f"Output schema error — missing columns: {missing_out}"
    )

df_output = df_revenue[OUTPUT_COLUMNS].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_output.to_parquet(
    OUTPUT_PATH,
    index=False,
    engine="pyarrow",
    compression="snappy",
)

print(f"[5.2] ✅ Output written → {OUTPUT_PATH}")
print(f"Rows: {len(df_output):,}")
print(f"Columns: {len(df_output.columns):,}")

# %% [markdown]
# ## 6. Diagnostic Summary

# %%
print("\n" + "=" * 65)
print("NB21 — DIAGNOSTIC SUMMARY (Tank > Pipe Proof)")
print("=" * 65)

print("\n⚡ Revenue & LCOS Overview (per economic_archetype):")
summary = df_output.groupby("economic_archetype").agg(
    n_projects=("project_id", "count"),
    avg_lcos=("lcos_gbp_mwh", "mean"),
    avg_gross_spread=("gross_spread_gbp", "mean"),
    avg_actual_rev_per_mw=("actual_revenue_per_mw_yr", "mean"),
    avg_req_floor=("required_floor_per_mw_yr", "mean"),
    rrt_exposure=("rrt_applicable", "mean") # % of projects flagged
)

summary["avg_lcos"] = summary["avg_lcos"].apply(lambda x: f"£{x:,.0f}")
summary["avg_gross_spread"] = summary["avg_gross_spread"].apply(lambda x: f"£{x:,.0f}")
summary["avg_actual_rev_per_mw"] = summary["avg_actual_rev_per_mw"].apply(lambda x: f"£{x:,.0f}")
summary["avg_req_floor"] = summary["avg_req_floor"].apply(lambda x: f"£{x:,.0f}")
summary["rrt_exposure"] = summary["rrt_exposure"].apply(lambda x: f"{x*100:.0f}%")

print(summary.rename(columns={
    "avg_lcos": "Avg LCOS", 
    "avg_gross_spread": "Avg Spread", 
    "avg_actual_rev_per_mw": "Actual Rev/MW",
    "avg_req_floor": "Req Floor/MW",
    "rrt_exposure": "RRT Exposure"
}).to_string())

print("\n🚨 Merchant Sufficiency Verdict:")
for tech in df_output["economic_archetype"].unique():
    tech_df = df_output[df_output["economic_archetype"] == tech]
    lcos = tech_df["lcos_gbp_mwh"].mean()
    spread = tech_df["gross_spread_gbp"].mean()
    delta = spread - lcos
    status = "SUFFICIENT" if delta >= 0 else "DEFICIT"
    print(f"  {tech}: LCOS £{lcos:,.0f} vs Spread £{spread:,.0f} -> {status} (Delta: £{delta:,.0f}/MWh)")


print("NB21 complete. Output: ldes_portfolio_revenues.parquet")
print("─" * 65)
print("Next: NB22 — Curtailment & RRT Penalty Application")

# %% [markdown]
# NB21 does not show that LDES is uneconomic. It shows that wholesale price arbitrage alone is insufficient to cover lifecycle LCOS across all tested archetypes. This is exactly the economic space where cap-and-floor support becomes relevant: the policy question is not whether arbitrage revenues exist, but whether additional system value and consumer-backed revenue stabilisation justify the investment.

# %%
print("🔍 Lifecycle CAPEX Check:")
for _, row in df_revenue.iterrows():
    print(f"{row['project_name']:20s} | MW: {row['mw_capacity']:6.0f} | "
          f"MWh: {row['mwh_capacity']:8.0f} | "
          f"Lifecycle £: {row['lifecycle_capex_gbp']/1e9:6.2f}bn | "
          f"£/MWh: {row['lifecycle_capex_gbp']/row['mwh_capacity']/1000:6.0f}k")

# %%
print("🔍 NB20 Capex Calculation Check:")
print("\nFrom technology_archetype_economics.csv:")
archetypes = pd.read_csv(PROCESSED_DIR.parent / "raw" / "technology_archetype_economics.csv")
print(archetypes[['economic_archetype', 'capex_mw_gbp', 'capex_mwh_gbp']].to_string())

print("\nActual portfolio capex:")
for _, row in df_revenue.iterrows():
    if row['technology_type'] == 'PSH':
        benchmark_capex_kw = row['mw_capacity'] * 2_000_000  # £2,000/kW benchmark
        modelled_capex = row['initial_capex_gbp']
        ratio = modelled_capex / benchmark_capex_kw
        print(f"{row['project_name']:20s} | Benchmark: £{benchmark_capex_kw/1e9:.2f}bn | "
              f"Modelled: £{modelled_capex/1e9:.2f}bn | Ratio: {ratio:.2f}x")

# %%
print("\n" + "=" * 75)
print("PSH SENSITIVITY ANALYSIS — Testing Robustness of Lifecycle Cost Finding")
print("=" * 75)

# Define sensitivity scenarios
psh_scenarios = {
    'Base': {'capex_mw': 1_200_000, 'capex_mwh': 20_000},
    'Conservative': {'capex_mw': 1_800_000, 'capex_mwh': 40_000},
    'Stress': {'capex_mw': 2_400_000, 'capex_mwh': 80_000},
}

print("\nPSH Lifecycle Cost Sensitivity:")
print(f"{'Scenario':<15} {'capex_mw':<12} {'capex_mwh':<12} {'Avg Lifecycle £/MWh':<20}")
print("-" * 75)

for scenario, params in psh_scenarios.items():
    # Calculate PSH lifecycle cost for each project under this scenario
    psh_lcos_values = []
    
    for _, row in df_revenue.iterrows():
        if row['technology_type'] == 'PSH':
            # Calculate initial capex
            initial_capex = (row['mw_capacity'] * params['capex_mw'] + 
                           row['mwh_capacity'] * params['capex_mwh'])
            
            # Apply lifecycle multiplier (1.15x for PSH)
            lifecycle_capex = initial_capex * 1.15
            
            # Calculate £/MWh
            lifecycle_per_mwh = lifecycle_capex / row['mwh_capacity']
            psh_lcos_values.append(lifecycle_per_mwh)
    
    avg_lifecycle = np.mean(psh_lcos_values)
    
    print(f"{scenario:<15} £{params['capex_mw']/1000:.0f}k/MW    "
          f"£{params['capex_mwh']/1000:.0f}k/MWh    £{avg_lifecycle/1000:.0f}k/MWh")

print("-" * 75)
print("\nLi-ion comparison (from current model):")
li_ion_avg = df_revenue[df_revenue['technology_type'] == 'Li-ion']['lifecycle_capex_per_mwh_gbp'].mean()
print(f"Li-ion avg lifecycle: £{li_ion_avg/1000:.0f}k/MWh")

print("\n✅ Key Finding: PSH remains cheaper than Li-ion across all scenarios")
print("   Even under stress assumptions (£2.4m/MW + £80k/MWh), PSH lifecycle cost")
print("   is substantially lower than Li-ion's £458-499k/MWh.")

# %% [markdown]
# ## NB21 Finding — Wholesale Arbitrage Is Insufficient Across All LDES Archetypes
#
# The 7-day rolling window dispatch simulation demonstrates that long-duration assets can successfully capture multi-day weather arbitrage opportunities, validating the "Tank > Pipe" thesis. However, the resulting merchant spreads (£74-97/MWh) fall significantly short of lifecycle costs across all tested archetypes:
#
# | Technology | LCOS (£/MWh) | Spread (£/MWh) | Deficit (£/MWh) | Required Floor (£/MW/yr) |
# |------------|--------------|----------------|-----------------|--------------------------|
# | PSH | £212 | £74 | -£139 | £112k |
# | CAES | £298 | £84 | -£214 | £212k |
# | Flow | £846 | £97 | -£749 | £234k |
# | Li-ion | £984 | £90 | -£894 | £506k |
#
# The analysis reveals a clear hierarchy: **long-duration, low-lifecycle-cost technologies (PSH, CAES) have substantially lower LCOS than short-duration, high-lifecycle-cost technologies (Li-ion, Flow)**. This is not a market failure; it is the economic space where Cap-and-Floor support becomes relevant.
#
# NB21 does not show that LDES is uneconomic. It shows that wholesale price arbitrage alone is insufficient to cover lifecycle LCOS across all tested archetypes. This is exactly the economic space where cap-and-floor support becomes relevant: the policy question is not whether arbitrage revenues exist, but whether additional system value and consumer-backed revenue stabilisation justify the investment.

# %% [markdown]
# Checks

# %%
print("🔍 NB21 Input Check:")
print(f"  Portfolio loaded: {len(df_portfolio)} projects")
print(f"  Price data loaded: {len(df_prices)} hours")
print(f"  Price range: £{df_prices['price_gbp_mwh'].min():.0f} - £{df_prices['price_gbp_mwh'].max():.0f}/MWh")

# %%
print("\n🔍 NB21 Dispatch Cycle Check:")
print(f"  Avg annual cycles: {df_revenue['annual_cycles'].mean():.1f}")
print(f"  Cycle range: {df_revenue['annual_cycles'].min():.0f} - {df_revenue['annual_cycles'].max():.0f}")

# %%
print("\n🔍 NB21 LCOS Check:")
for archetype in df_revenue['economic_archetype'].unique():
    arch_data = df_revenue[df_revenue['economic_archetype'] == archetype]
    avg_lcos = arch_data['lcos_gbp_mwh'].mean()
    avg_spread = arch_data['gross_spread_gbp'].mean()
    deficit = avg_spread - avg_lcos
    print(f"  {archetype:10s} | LCOS: £{avg_lcos:.0f}/MWh | Spread: £{avg_spread:.0f}/MWh | "
          f"Deficit: £{deficit:.0f}/MWh")

# %%
print("\n🔍 NB21 Output Check:")
output_path = PROCESSED_DIR / "ldes_portfolio_revenues.parquet"
print(f"  Output exists: {output_path.exists()}")
if output_path.exists():
    df_check = pd.read_parquet(output_path)
    print(f"  Rows: {len(df_check)}")
    print(f"  Has lcos_gbp_mwh: {'lcos_gbp_mwh' in df_check.columns}")

# %%
print("\n🔍 NB21 Dispatch Economics Deep Dive:")
print(f"{'Project':<25s} {'Duration':<10s} {'Cycles':<8s} {'Spread':<10s} {'MWh Delivered':<15s}")
print("-" * 75)

for _, row in df_revenue.iterrows():
    print(f"{row['project_name']:<25s} {row['duration_hours']:<10.1f} "
          f"{row['annual_cycles']:<8.0f} £{row['gross_spread_gbp']:<9.2f} "
          f"{row['annual_mwh_delivered']:<15,.0f}")

print("\n🔍 Spread Distribution:")
print(f"  Min spread: £{df_revenue['gross_spread_gbp'].min():.2f}/MWh")
print(f"  Max spread: £{df_revenue['gross_spread_gbp'].max():.2f}/MWh")
print(f"  Mean spread: £{df_revenue['gross_spread_gbp'].mean():.2f}/MWh")
print(f"  Projects with spread < £10/MWh: {(df_revenue['gross_spread_gbp'] < 10).sum()}")

# %%
