# %% [markdown]
# # NB23 — Stacked Revenue & Required Floor
# **Phase 2 — GB Constraint Analytics Platform**
#
# **Scope:** Add non-wholesale revenue streams (Capacity Market, Balancing Mechanism) 
# to calculate the Residual Floor Requirement — the exact gap Cap-and-Floor must fill.
#
# **Methodological Corrections applied here:**
# -  **Correction 8 (Duration-Scaled Capacity Market):** CM payments scale with 
#   duration. A 4h asset provides less system security than a 24h asset.
# -  **Correction 9 (The Revenue Waterfall):** We do not blend revenues. We 
#   calculate a strict waterfall: LCOS -> Wholesale Spread -> Curtailment Penalty 
#   -> Stacked Revenue -> Residual Floor.

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

NB22_INPUT_PATH = PROCESSED_DIR / "ldes_portfolio_curtailment.parquet"
OUTPUT_PATH = OUTPUT_DIR / "ldes_portfolio_stacked.parquet"

print("NB23 — Stacked Revenue & Required Floor")
print("=" * 65)

# %% [markdown]
# ## 1. Data Ingestion

# %%
def load_portfolio(path: Path) -> pd.DataFrame:
    print(f"\n[1.1] Loading NB22 curtailment portfolio...")
    df = pd.read_parquet(path)
    print(f"  ✅ {len(df)} projects loaded.")
    return df

df_portfolio = load_portfolio(NB22_INPUT_PATH)

# %% [markdown]
# ## 2. Correction 8: Duration-Scaled Stacked Revenues

# %%
def calculate_stacked_revenues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Capacity Market and Balancing Mechanism revenues.
    """
    print("\n[2.1] Calculating Duration-Scaled Stacked Revenues (Correction 8)...")
    
    # 1. Capacity Market (CM) - Scales with duration
    # 4h asset: £30k/MW/yr
    # 8h asset: £50k/MW/yr
    # 24h+ asset: £80k/MW/yr
    # Formula: Linear interpolation between 4h and 24h, capped at 24h.
    cm_base = 30_000  # £/MW/yr for 4h
    cm_max = 80_000   # £/MW/yr for 24h+
    
    duration_factor = np.minimum(1.0, (df['duration_hours'] - 4) / (24 - 4))
    duration_factor = np.maximum(0.0, duration_factor) # Floor at 0 for <4h
    
    df['cm_revenue_per_mw_yr'] = cm_base + (duration_factor * (cm_max - cm_base))
    
    # 2. Balancing Mechanism (BM) & Ancillary Services
    # Flat rate for providing flexibility, regardless of duration.
    # Typical range: £20k - £30k/MW/yr. We use £25k.
    bm_rate = 25_000
    df['bm_revenue_per_mw_yr'] = bm_rate
    
    # 3. Total Stacked Revenue
    df['total_stacked_revenue_per_mw_yr'] = (
        df['cm_revenue_per_mw_yr'] + 
        df['bm_revenue_per_mw_yr']
    )
    
    print(f"  ✅ Stacked revenues calculated.")
    return df

df_stacked = calculate_stacked_revenues(df_portfolio)

# %% [markdown]
# ## 3. Correction 9: The Revenue Waterfall & Residual Floor

# %%
def calculate_residual_floor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Residual Floor Requirement.
    This is the exact gap the Cap-and-Floor regime must fill.
    """
    print("\n[3.1] Calculating Residual Floor Requirement (Correction 9)...")
    
    # The Adjusted Required Floor from NB22 is the total revenue needed to cover LCOS.
    # We subtract the stacked revenues to find the residual gap.
    
    df['residual_floor_per_mw_yr'] = np.maximum(
        0, 
        df['required_floor_per_mw_yr_adj'] - df['total_stacked_revenue_per_mw_yr']
    )
    
    # Calculate the "Merchant Sufficiency Ratio" (Stacked Revenue / Required Floor)
    # If > 1.0, the project is viable without Cap-and-Floor.
    df['merchant_sufficiency_ratio'] = np.where(
        df['required_floor_per_mw_yr_adj'] > 0,
        df['total_stacked_revenue_per_mw_yr'] / df['required_floor_per_mw_yr_adj'],
        1.0 # If no floor needed, ratio is 1.0 (100% sufficient)
    )
    
    print(f"  ✅ Residual floor calculated.")
    return df

df_stacked = calculate_residual_floor(df_stacked)

# %% [markdown]
# ## 4. Output Construction & Export

# %%
NEW_COLS = [
    'cm_revenue_per_mw_yr', 'bm_revenue_per_mw_yr', 
    'total_stacked_revenue_per_mw_yr', 'residual_floor_per_mw_yr',
    'merchant_sufficiency_ratio'
]

OUTPUT_COLUMNS = df_stacked.columns.tolist()
df_output = df_stacked[OUTPUT_COLUMNS].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_output.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")

print(f"\n[4.1] ✅ Output written → {OUTPUT_PATH}")

# %% [markdown]
# ## 5. Diagnostic Summary

# %%
print("\n" + "=" * 65)
print("NB23 — DIAGNOSTIC SUMMARY (Revenue Waterfall)")
print("=" * 65)

print("\n💰 Revenue Waterfall Overview (per economic_archetype):")
summary = df_output.groupby("economic_archetype").agg(
    n_projects=("project_id", "count"),
    avg_duration=("duration_hours", "mean"),
    avg_cm=("cm_revenue_per_mw_yr", "mean"),
    adj_floor=("required_floor_per_mw_yr_adj", "mean"),
    residual_floor=("residual_floor_per_mw_yr", "mean"),
    sufficiency=("merchant_sufficiency_ratio", "mean")
)

summary["avg_duration"] = summary["avg_duration"].apply(lambda x: f"{x:.0f}h")
summary["avg_cm"] = summary["avg_cm"].apply(lambda x: f"£{x/1000:.0f}k")
summary["adj_floor"] = summary["adj_floor"].apply(lambda x: f"£{x/1000:.0f}k")
summary["residual_floor"] = summary["residual_floor"].apply(lambda x: f"£{x/1000:.0f}k")
summary["sufficiency"] = summary["sufficiency"].apply(lambda x: f"{x*100:.0f}%")

print(summary.rename(columns={
    "avg_duration": "Avg Duration",
    "avg_cm": "CM Revenue",
    "adj_floor": "Adj Req Floor",
    "residual_floor": "Residual Floor",
    "sufficiency": "Stacked Sufficiency"
}).to_string())

print("\n🚨 The Cap-and-Floor Gap:")
for tech in df_output["economic_archetype"].unique():
    tech_df = df_output[df_output["economic_archetype"] == tech]
    adj_floor = tech_df["required_floor_per_mw_yr_adj"].mean()
    stacked = tech_df["total_stacked_revenue_per_mw_yr"].mean()
    residual = tech_df["residual_floor_per_mw_yr"].mean()
    
    print(f"  {tech}: Needs £{adj_floor/1000:.0f}k/MW/yr | Stacked provides £{stacked/1000:.0f}k/MW/yr | Cap-and-Floor must fill £{residual/1000:.0f}k/MW/yr")

print("\n" + "─" * 65)
print("NB23 complete. Output: ldes_portfolio_stacked.parquet")
print("─" * 65)
print("Next: NB24 — Consumer Exposure & Support Justification Ratio")

# %% [markdown]
# ## NB23 Finding — Stacked Revenues Partially Close the Gap, but Technology-Specific Support Remains Essential
#
# The addition of Capacity Market and Balancing Mechanism revenues creates a "Revenue Waterfall" that partially closes the merchant sufficiency gap established in NB21 and NB22. However, substantial residual floor requirements remain for most archetypes.
#
# **The Revenue Waterfall:**
#
# | Technology | Avg Duration | CM Revenue | BM Revenue | Total Stacked | Adj Required Floor | Residual Floor | Stacked Sufficiency |
# |------------|--------------|------------|------------|---------------|-------------------|----------------|---------------------|
# | PSH | 23h | £71k | £25k | £96k | £117k | £21k | 83% |
# | CAES | 30h | £80k | £25k | £105k | £212k | £107k | 49% |
# | Flow | 8h | £40k | £25k | £65k | £234k | £169k | 28% |
# | Li-ion | 13h | £53k | £25k | £78k | £511k | £433k | 16% |
#
# **Key Findings:**
#
# 1. **PSH is closest to merchant viability.** With 83% stacked sufficiency, PSH requires only £21k/MW/yr of Cap-and-Floor support. Its combination of low lifecycle cost, long duration (enabling high CM revenue), and duration resilience (mitigating curtailment) makes it the most commercially robust LDES technology in the Window 1 portfolio.
#
# 2. **Duration drives Capacity Market value.** The CM revenue formula scales with duration: longer-duration assets provide more system security during winter peaks and receive higher payments. PSH (23h) and CAES (30h) capture £71k and £80k/MW/yr respectively, while Flow (8h) captures only £40k/MW/yr. This creates a structural advantage for long-duration technologies in stacked revenue models.
#
# 3. **Li-ion requires the most support by a wide margin.** At £433k/MW/yr residual floor, Li-ion needs **20x more support than PSH** and **4x more than Flow**. This is driven by the combination of high lifecycle cost (£1,088/MWh adjusted LCOS) and moderate duration (13h average), which limits both CM revenue and multi-day arbitrage capture.
#
# 4. **The Li-ion internal split matters.** The 11 Li-ion projects are not homogeneous. The 4 Field North Scotland projects (16-18h duration, RRT-exposed) face curtailment penalties that increase their residual floor, while the 7 southern projects (8-12h, unconstrained) have lower requirements. Cap-and-Floor design must account for this location-specific variation.
#
# **The Central Finding:** Stacked revenues improve project bankability but do not eliminate the need for Cap-and-Floor support across most archetypes. The residual floor requirement varies by **20x** across technologies (from £21k/MW/yr for PSH to £433k/MW/yr for Li-ion), demonstrating that a uniform Cap-and-Floor level would over-support low-cost technologies and under-support high-cost ones.
#
# NB23 does not show that stacked revenues are sufficient. It shows that they partially bridge the gap, but technology-specific support levels are essential to reflect the true cost structure and system value of each archetype.

# %% [markdown]
# Check

# %%
print("🔍 NB23 Input Check:")
print(f"  Portfolio loaded: {len(df_portfolio)} projects")
print(f"  Has curtailment-adjusted values: {'lcos_gbp_mwh_adj' in df_portfolio.columns}")

# %%
print("\n🔍 NB23 Capacity Market Check:")
for archetype in df_stacked['economic_archetype'].unique():
    arch_data = df_stacked[df_stacked['economic_archetype'] == archetype]
    avg_cm = arch_data['cm_revenue_per_mw_yr'].mean()
    avg_duration = arch_data['duration_hours'].mean()
    print(f"  {archetype:10s} | Avg Duration: {avg_duration:.1f}h | CM Revenue: £{avg_cm/1000:.0f}k/MW/yr")

# %%
print("\n🔍 NB23 Residual Floor Check:")
for archetype in df_stacked['economic_archetype'].unique():
    arch_data = df_stacked[df_stacked['economic_archetype'] == archetype]
    adj_floor = arch_data['required_floor_per_mw_yr_adj'].mean()
    residual = arch_data['residual_floor_per_mw_yr'].mean()
    sufficiency = arch_data['merchant_sufficiency_ratio'].mean()
    print(f"  {archetype:10s} | Adj Floor: £{adj_floor/1000:.0f}k | Residual: £{residual/1000:.0f}k | "
          f"Sufficiency: {sufficiency*100:.0f}%")

# %%
print("\n🔍 NB23 Output Check:")
output_path = PROCESSED_DIR / "ldes_portfolio_stacked.parquet"
print(f"  Output exists: {output_path.exists()}")
if output_path.exists():
    df_check = pd.read_parquet(output_path)
    print(f"  Rows: {len(df_check)}")
    print(f"  Has residual_floor_per_mw_yr: {'residual_floor_per_mw_yr' in df_check.columns}")
