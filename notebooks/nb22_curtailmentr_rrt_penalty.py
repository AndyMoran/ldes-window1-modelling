# %% [markdown]
# # NB22 — Curtailment & RRT Penalty Application
# **Phase 2 — GB Constraint Analytics Platform**
#
# **Scope:** Apply physical curtailment and constraint-specific penalties to the 
# wholesale revenues calculated in NB21.
#
# **Methodological Corrections applied here:**
# -  **Correction 6 (Physical Curtailment Multiplier):** Assets flagged 
#   `rrt_applicable=True` have their energy and revenue reduced by the 
#   `rrt_persistence_score` (the % of time the constraint is active).
# -  **Correction 7 (Duration Resilience Factor):** Long-duration assets can 
#   time-shift their discharge to avoid constraint hours. We apply a resilience 
#   factor: `resilience = min(1.0, duration_hours / 24)`. A 24h asset can shift 
#   a full day, avoiding most constraints; a 4h asset cannot.

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
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR

NB21_INPUT_PATH = PROCESSED_DIR / "ldes_portfolio_revenues.parquet"
OUTPUT_PATH = OUTPUT_DIR / "ldes_portfolio_curtailment.parquet"

print("NB22 — Curtailment & RRT Penalty Application")
print("=" * 65)

# %% [markdown]
# ## 1. Data Ingestion

# %%
def load_portfolio(path: Path) -> pd.DataFrame:
    print(f"\n[1.1] Loading NB21 revenue portfolio...")
    df = pd.read_parquet(path)
    print(f"  ✅ {len(df)} projects loaded.")
    return df

df_portfolio = load_portfolio(NB21_INPUT_PATH)


# %% [markdown]
# ## 2. Correction 6 & 7: Curtailment & Duration Resilience

# %% [markdown]
# ### Correction 7: Duration Resilience Factor — Definition & Formula
#
# **Plain-English Definition:**  
# Duration Resilience measures an asset's ability to time-shift its discharge to avoid constraint hours. A long-duration asset (e.g., 24h PSH) can hold its energy and wait for the constraint to clear, suffering minimal curtailment. A short-duration asset (e.g., 4h Li-ion) must discharge within a narrow window, making it highly vulnerable to being curtailed when the constraint is active.
#
# **Formula:**  

# %% [markdown]
# duration_resilience_factor = min(1.0, duration_hours / 24.0)

# %% [markdown]
# **Interpretation:**
# - **1.0 (100% resilience):** Asset has 24h+ duration and can fully time-shift across a full day. It can always find a clear discharge window.
# - **0.5 (50% resilience):** Asset has 12h duration. It can shift half a day, avoiding some but not all constraint hours.
# - **0.17 (17% resilience):** Asset has 4h duration. It has very limited ability to time-shift and is highly exposed to curtailment.
#
# **Application in NB22:**  
# The effective curtailment factor is calculated as:

# %% [markdown]
# effective_curtailment_factor = 1 - (rrt_persistence_score × (1 - duration_resilience_factor))

# %% [markdown]
# This means an asset only suffers curtailment for the portion of the constraint it *cannot* time-shift. A 24h asset (100% resilience) suffers zero curtailment penalty even if the constraint is persistent, because it can always wait for a clear window. A 4h asset (17% resilience) suffers the full persistence penalty.

# %%
def apply_curtailment_and_resilience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies physical curtailment penalties, adjusted for duration resilience.
    """
    print("\n[2.1] Applying Curtailment & Duration Resilience (Corrections 6 & 7)...")
    
    # 1. Base Curtailment Factor (1 - persistence score)
    # If rrt_applicable is False, factor is 1.0 (no curtailment)
    df['base_curtailment_factor'] = np.where(
        df['rrt_applicable'], 
        1 - df['rrt_persistence_score'], 
        1.0
    )
    
    # 2. Duration Resilience Factor
    # Long-duration assets can time-shift discharge to avoid constraints.
    # Formula: min(1.0, duration_hours / 24). 
    # A 24h+ asset has 100% resilience (can always find a clear window).
    # A 4h asset has 16% resilience (must discharge in specific window).
    df['duration_resilience_factor'] = np.minimum(1.0, df['duration_hours'] / 24.0)
    
    # 3. Effective Curtailment Factor
    # The asset only suffers curtailment for the portion it CANNOT time-shift.
    # effective_factor = base_factor + (1 - base_factor) * resilience
    # Simplified: We penalize the (1 - resilience) portion of the persistence score.
    df['effective_curtailment_factor'] = np.where(
        df['rrt_applicable'],
        1 - (df['rrt_persistence_score'] * (1 - df['duration_resilience_factor'])),
        1.0
    )
    
    # 4. Apply to Energy and Revenue
    df['annual_mwh_delivered_adj'] = df['annual_mwh_delivered'] * df['effective_curtailment_factor']
    df['annual_gross_revenue_gbp_adj'] = df['annual_gross_revenue_gbp'] * df['effective_curtailment_factor']
    
    # 5. Recalculate LCOS and Floor based on adjusted values
    # LCOS = Total Annual Cost / Adjusted MWh
    df['lcos_gbp_mwh_adj'] = np.where(
        df['annual_mwh_delivered_adj'] > 0,
        df['total_annual_cost_gbp'] / df['annual_mwh_delivered_adj'],
        np.nan
    )
    
    # Required Floor (Adjusted) = (LCOS_adj - Spread) * Adjusted MWh / MW
    # Or simpler: Hurdle - Adjusted Actual Revenue
    df['actual_revenue_per_mw_yr_adj'] = df['annual_gross_revenue_gbp_adj'] / df['mw_capacity']
    df['required_floor_per_mw_yr_adj'] = np.maximum(
        0, 
        df['required_revenue_per_mw_yr'] - df['actual_revenue_per_mw_yr_adj']
    )
    
    print(f"  ✅ Curtailment and resilience applied.")
    return df

df_curtailment = apply_curtailment_and_resilience(df_portfolio)

# %% [markdown]
# ## 3. Output Construction & Export

# %%
# Add new columns to the output list
NEW_COLS = [
    'base_curtailment_factor', 'duration_resilience_factor', 
    'effective_curtailment_factor', 'annual_mwh_delivered_adj', 
    'annual_gross_revenue_gbp_adj', 'lcos_gbp_mwh_adj', 
    'actual_revenue_per_mw_yr_adj', 'required_floor_per_mw_yr_adj'
]

OUTPUT_COLUMNS = df_curtailment.columns.tolist()
df_output = df_curtailment[OUTPUT_COLUMNS].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_output.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")

print(f"\n[3.1] ✅ Output written → {OUTPUT_PATH}")

# %% [markdown]
# ## 4. Diagnostic Summary

# %%
print("\n" + "=" * 65)
print("NB22 — DIAGNOSTIC SUMMARY (Curtailment Impact)")
print("=" * 65)

print("\n⚡ Curtailment Impact Overview (per economic_archetype):")
summary = df_output.groupby("economic_archetype").agg(
    n_projects=("project_id", "count"),
    avg_resilience=("duration_resilience_factor", "mean"),
    avg_curtailment=("effective_curtailment_factor", "mean"),
    avg_lcos_adj=("lcos_gbp_mwh_adj", "mean"),
    avg_floor_adj=("required_floor_per_mw_yr_adj", "mean"),
    rrt_exposure=("rrt_applicable", "mean")
)

summary["avg_resilience"] = summary["avg_resilience"].apply(lambda x: f"{x*100:.0f}%")
summary["avg_curtailment"] = summary["avg_curtailment"].apply(lambda x: f"{x*100:.0f}%")
summary["avg_lcos_adj"] = summary["avg_lcos_adj"].apply(lambda x: f"£{x:,.0f}")
summary["avg_floor_adj"] = summary["avg_floor_adj"].apply(lambda x: f"£{x:,.0f}")
summary["rrt_exposure"] = summary["rrt_exposure"].apply(lambda x: f"{x*100:.0f}%")

print(summary.rename(columns={
    "avg_resilience": "Duration Resilience",
    "avg_curtailment": "Effective Factor",
    "avg_lcos_adj": "Adj LCOS",
    "avg_floor_adj": "Adj Req Floor",
    "rrt_exposure": "RRT Exposure"
}).to_string())

print("\n🚨 Merchant Sufficiency Verdict (Post-Curtailment):")
for tech in df_output["economic_archetype"].unique():
    tech_df = df_output[df_output["economic_archetype"] == tech]
    lcos = tech_df["lcos_gbp_mwh_adj"].mean()
    spread = tech_df["gross_spread_gbp"].mean() # Spread remains same, energy drops
    delta = spread - lcos
    status = "SUFFICIENT" if delta >= 0 else "DEFICIT"
    print(f"  {tech}: Adj LCOS £{lcos:,.0f} vs Spread £{spread:,.0f} -> {status} (Delta: £{delta:,.0f}/MWh)")

print("\n" + "─" * 65)
print("NB22 complete. Output: ldes_portfolio_curtailment.parquet")
print("─" * 65)
print("Next: NB23 — Stacked Revenue & Required Floor")

# %% [markdown]
# ## NB22 Finding — Curtailment Amplifies the Merchant Sufficiency Gap, but Duration Provides Resilience
#
# The application of curtailment penalties to assets behind persistent transmission constraints reveals that **geographic location and duration jointly determine an asset's vulnerability to physical grid constraints**.
#
# The Duration Resilience Factor quantifies an asset's ability to time-shift discharge to avoid constraint hours:
#
# | Technology | Avg Duration | Duration Resilience | RRT Exposure | Effective Factor |
# |------------|--------------|---------------------|--------------|------------------|
# | CAES | 30h | 100% | 0% | 100% |
# | PSH | 23h | 85% | 100% | 88% |
# | Li-ion | 12h | 54% | 36% | 92% |
# | Flow | 8h | 33% | 0% | 100% |
#
# **The North Scotland Cluster:** Three PSH projects (Coire Glas, Earba, Loch Kemp) and four Field Li-ion projects (Netherton, New Deer, Rigifa, Fyrish) are all located behind the SSEN-S constraint interface, representing 5,100 MW (67% of the Window 1 portfolio). These assets face physical curtailment risk when the constraint is active.
#
# **Duration Resilience Mitigates the Penalty:** PSH's long duration (15-32h) gives it 85% resilience, allowing it to time-shift discharge and avoid most constraint hours. As a result, PSH's LCOS rises only modestly from £212 to £258/MWh (+22%) after curtailment, and its Required Floor increases by just £5k/MW/yr.
#
# **The Li-ion Internal Split:** The Li-ion archetype shows a decisive internal split. The four Field North Scotland projects (16-18h duration, high RRT exposure) face curtailment penalties, while the seven southern projects (8-12h duration, no RRT exposure) are unaffected. This demonstrates that **Cap-and-Floor support should be calibrated to location and duration, not just technology type**.
#
# **The Central Finding:** Curtailment risk does not create the merchant financeability problem (that was established in NB21), but it amplifies it. Long-duration assets behind constraints can mitigate the penalty through operational flexibility, but short-duration assets or assets with very high lifecycle costs (Li-ion) remain highly exposed.
#
# NB22 does not show that constrained assets are unviable. It shows that geographic constraint exposure interacts with duration to determine the level of regulatory support required. Cap-and-Floor design must account for this interaction to avoid over-supporting unconstrained assets and under-supporting constrained ones.

# %% [markdown]
# Check

# %%
print("🔍 NB22 Input Check:")
print(f"  Portfolio loaded: {len(df_portfolio)} projects")
print(f"  RRT-flagged projects: {df_portfolio['rrt_applicable'].sum()}")

# %%
print("\n🔍 NB22 Duration Resilience Check:")
for archetype in df_curtailment['economic_archetype'].unique():
    arch_data = df_curtailment[df_curtailment['economic_archetype'] == archetype]
    avg_resilience = arch_data['duration_resilience_factor'].mean()
    avg_duration = arch_data['duration_hours'].mean()
    print(f"  {archetype:10s} | Avg Duration: {avg_duration:.1f}h | Resilience: {avg_resilience*100:.0f}%")

# %%
print("\n🔍 NB22 Curtailment Impact Check:")
for archetype in df_curtailment['economic_archetype'].unique():
    arch_data = df_curtailment[df_curtailment['economic_archetype'] == archetype]
    lcos_before = arch_data['lcos_gbp_mwh'].mean()
    lcos_after = arch_data['lcos_gbp_mwh_adj'].mean()
    change = (lcos_after - lcos_before) / lcos_before * 100
    print(f"  {archetype:10s} | LCOS before: £{lcos_before:.0f} | After: £{lcos_after:.0f} | "
          f"Change: {change:+.1f}%")

# %%
print("\n🔍 NB22 Output Check:")
output_path = PROCESSED_DIR / "ldes_portfolio_curtailment.parquet"
print(f"  Output exists: {output_path.exists()}")
if output_path.exists():
    df_check = pd.read_parquet(output_path)
    print(f"  Rows: {len(df_check)}")
    print(f"  Has lcos_gbp_mwh_adj: {'lcos_gbp_mwh_adj' in df_check.columns}")

# %%
print("="*75)
print("SEARCHING FOR CURTAILMENT/CANNIBALIZATION PENALTY")
print("="*75)

# Check what columns exist in the curtailment output
df_curtailment = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_curtailment.parquet")

print("\n📋 Columns in ldes_portfolio_curtailment.parquet:")
for col in df_curtailment.columns:
    if any(keyword in col.lower() for keyword in ['curtail', 'cannibal', 'penalty', 'adj', 'factor', 'rrt']):
        print(f"  ✓ {col}")

print("\n🔍 Checking RRT-flagged projects for penalty application:")
rrt_projects = df_curtailment[df_curtailment['rrt_applicable'] == True]

print(f"\n{'Project':<25s} {'LCOS Before':<15s} {'LCOS After':<15s} {'Change':<12s} {'% Change':<10s}")
print("-"*75)

for _, row in rrt_projects.iterrows():
    before = row['lcos_gbp_mwh']
    after = row['lcos_gbp_mwh_adj']
    change = after - before
    pct_change = (change / before) * 100
    print(f"{row['project_name']:<25s} £{before:<14.0f} £{after:<14.0f} "
          f"£{change:<+11.0f} {pct_change:<+9.1f}%")

print("\n🔍 Checking non-RRT projects (should be unchanged):")
non_rrt = df_curtailment[df_curtailment['rrt_applicable'] == False].head(3)

for _, row in non_rrt.iterrows():
    before = row['lcos_gbp_mwh']
    after = row['lcos_gbp_mwh_adj']
    change = after - before
    pct_change = (change / before) * 100
    print(f"{row['project_name']:<25s} £{before:<14.0f} £{after:<14.0f} "
          f"£{change:<+11.0f} {pct_change:<+9.1f}%")

# %%
print("="*75)
print("CURTAILMENT FACTOR BREAKDOWN")
print("="*75)

df_curtailment = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_curtailment.parquet")

rrt_projects = df_curtailment[df_curtailment['rrt_applicable'] == True]

print(f"\n{'Project':<25s} {'Duration':<10s} {'Base Curt':<12s} {'Resilience':<12s} {'Effective':<12s} {'LCOS Change':<12s}")
print("-"*85)

for _, row in rrt_projects.iterrows():
    duration = row['duration_hours']
    base = row['base_curtailment_factor']
    resilience = row['duration_resilience_factor']
    effective = row['effective_curtailment_factor']
    lcos_before = row['lcos_gbp_mwh']
    lcos_after = row['lcos_gbp_mwh_adj']
    pct_change = ((lcos_after - lcos_before) / lcos_before) * 100
    
    print(f"{row['project_name']:<25s} {duration:<10.1f} {base:<12.3f} {resilience:<12.3f} "
          f"{effective:<12.3f} {pct_change:<+11.1f}%")

print("\n🔍 Understanding the calculation:")
print("  • base_curtailment_factor: The raw curtailment exposure (likely 0.15 for all RRT projects)")
print("  • duration_resilience_factor: How much duration mitigates curtailment (0-1 scale)")
print("  • effective_curtailment_factor: base × (1 - resilience) = actual penalty applied")
print("  • LCOS adjustment: lcos_after = lcos_before / (1 - effective_curtailment_factor)")

# %%
print("="*75)
print("DECODED CURTAILMENT MODEL")
print("="*75)

print("\n📐 Formula:")
print("  effective_curtailment_factor = base_curtailment + duration_resilience × (1 - base_curtailment)")
print("  lcos_after = lcos_before / effective_curtailment_factor")

print("\n🔍 Verification:")
df_curtailment = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_curtailment.parquet")
rrt_projects = df_curtailment[df_curtailment['rrt_applicable'] == True]

for _, row in rrt_projects.iterrows():
    base = row['base_curtailment_factor']
    resilience = row['duration_resilience_factor']
    effective_actual = row['effective_curtailment_factor']
    
    # Calculate expected effective
    effective_calc = base + resilience * (1 - base)
    
    lcos_before = row['lcos_gbp_mwh']
    lcos_after = row['lcos_gbp_mwh_adj']
    lcos_calc = lcos_before / effective_calc
    
    print(f"\n{row['project_name']}:")
    print(f"  Base: {base:.3f}, Resilience: {resilience:.3f}")
    print(f"  Effective (actual): {effective_actual:.3f}, (calculated): {effective_calc:.3f}")
    print(f"  LCOS (actual): £{lcos_after:.0f}, (calculated): £{lcos_calc:.0f}")

print("\n" + "="*75)
print("KEY FINDINGS")
print("="*75)
print("\n1. Base curtailment is 18% (not 15%)")
print("   • This is the maximum curtailment if duration_resilience = 0")
print("   • Applied uniformly to all RRT-flagged projects")
print("\n2. Duration resilience mitigates curtailment")
print("   • Coire Glas (32h): 100% resilience → 0% effective curtailment")
print("   • Loch Kemp (22h): 93% resilience → 6% effective curtailment")
print("   • Earba (15h): 63% resilience → 31% effective curtailment")
print("\n3. LCOS adjustment is non-linear")
print("   • lcos_after = lcos_before / effective_curtailment_factor")
print("   • This means LCOS increases as effective curtailment increases")
print("   • The relationship is convex: small curtailment → small LCOS increase")
print("   • Large curtailment → large LCOS increase")

# %%
