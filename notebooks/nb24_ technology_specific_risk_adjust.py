# %% [markdown]
# # NB24 — Technology-Specific Risk Adjustment & Support Justification
# **Phase 2 — GB Constraint Analytics Platform**
#
# **Scope:** Stress-test the NB23 baseline by applying technology-specific risk adjustments.
# Tests whether CAES viability is robust to realistic geological and delivery-risk premia.
#
# **Methodological Corrections applied here:**
# - **Correction 10 (Multi-Channel Risk Adjustment):** Applies simultaneous adjustments 
#   to WACC, capex contingency, revenue capture, and availability to model realistic 
#   project finance risk premia.
# - **Correction 11 (Support Justification Ratio):** Calculates the SJR to determine 
#   whether proposed Cap-and-Floor levels are adequate.

# %% [markdown]
# ## Executive Summary
#
# *Lead Finding — Duration is the Primary Calibrator:*
#
# Our analysis of the 16 Window 1 LDES projects reveals that **duration is the primary driver of residual floor requirements**, with each additional hour of duration adding approximately £27.8k/MW/yr to the required Cap-and-Floor level (R² = 0.999, p < 0.0001). This continuous, linear relationship provides Ofgem with a clear, policy-actionable calibration mechanism: Cap-and-Floor should track duration, not just technology type.
#
# *Supporting Finding — Geographic Premium is Substantial:*
#
# Li-ion projects located behind the SSEN-S constraint interface require approximately 30% more support than unconstrained projects, after controlling for duration. This demonstrates that location-specific calibration is essential, though the magnitude may be conservative if partial Balancing Mechanism redispatch revenue is available during constraint events.
#
# *Supporting Finding — PSH Requires Minimal Incremental Support:*
#
# PSH projects demonstrate near-zero residual floor requirements under central assumptions, indicating they require materially less support than Li-ion. However, this finding is sensitive to construction-period financing risk (Interest During Construction) that is not captured in the current model, and developer statements suggest commercial uncertainties beyond pure cost metrics.
#
# *Policy Recommendation:*
#
# Cap-and-Floor should be calibrated primarily to duration (£27.8k/MW/yr per hour), with secondary adjustments for geographic constraint exposure and technology-specific lifecycle costs. A uniform floor would simultaneously over-support resilient long-duration assets and under-support constrained shorter-duration assets, distorting the optimal technology mix.

# %%
## 0. Imports and Configuration
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.2f}".format)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR

NB23_INPUT_PATH = PROCESSED_DIR / "ldes_portfolio_stacked.parquet"
OUTPUT_PATH = OUTPUT_DIR / "ldes_portfolio_risk_adjusted.parquet"

print("NB24 — Technology-Specific Risk Adjustment & Support Justification")
print("=" * 75)

# %% [markdown]
# ## 1. Data Ingestion

# %%
def load_portfolio(path: Path) -> pd.DataFrame:
    print(f"\n[1.1] Loading NB23 stacked revenue portfolio...")
    df = pd.read_parquet(path)
    print(f"  ✅ {len(df)} projects loaded.")
    return df

df_baseline = load_portfolio(NB23_INPUT_PATH)

# %% [markdown]
# ## 2. Correction 10: Multi-Channel Risk Adjustment Scenarios

# %%
def apply_risk_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies three scenarios: Base (NB23), Risk-adjusted, and Stress.
    Adjusts WACC, capex, revenue capture, and availability.
    """
    print("\n[2.1] Applying Technology-Specific Risk Scenarios (Correction 10)...")
    
    # Define risk parameters for each archetype and scenario
    # Format: {archetype: {scenario: {parameter: adjustment}}}
    risk_params = {
        'CAES': {
            'Base': {'wacc_adj': 0.0, 'capex_adj': 0.0, 'revenue_adj': 0.0, 'availability': 1.0},
            'Risk-adjusted': {'wacc_adj': 0.02, 'capex_adj': 0.15, 'revenue_adj': -0.10, 'availability': 0.95},
            'Stress': {'wacc_adj': 0.04, 'capex_adj': 0.30, 'revenue_adj': -0.20, 'availability': 0.90}
        },
        'PSH': {
            'Base': {'wacc_adj': 0.0, 'capex_adj': 0.0, 'revenue_adj': 0.0, 'availability': 1.0},
            'Risk-adjusted': {'wacc_adj': 0.01, 'capex_adj': 0.10, 'revenue_adj': -0.05, 'availability': 0.97},
            'Stress': {'wacc_adj': 0.02, 'capex_adj': 0.20, 'revenue_adj': -0.10, 'availability': 0.94}
        },
        'Li-ion': {
            'Base': {'wacc_adj': 0.0, 'capex_adj': 0.0, 'revenue_adj': 0.0, 'availability': 1.0},
            'Risk-adjusted': {'wacc_adj': 0.01, 'capex_adj': 0.05, 'revenue_adj': -0.05, 'availability': 0.97},
            'Stress': {'wacc_adj': 0.02, 'capex_adj': 0.10, 'revenue_adj': -0.10, 'availability': 0.94}
        },
        'Flow': {
            'Base': {'wacc_adj': 0.0, 'capex_adj': 0.0, 'revenue_adj': 0.0, 'availability': 1.0},
            'Risk-adjusted': {'wacc_adj': 0.015, 'capex_adj': 0.15, 'revenue_adj': -0.10, 'availability': 0.95},
            'Stress': {'wacc_adj': 0.03, 'capex_adj': 0.25, 'revenue_adj': -0.15, 'availability': 0.90}
        },
        'LAES': {
            'Base': {'wacc_adj': 0.0, 'capex_adj': 0.0, 'revenue_adj': 0.0, 'availability': 1.0},
            'Risk-adjusted': {'wacc_adj': 0.01, 'capex_adj': 0.10, 'revenue_adj': -0.05, 'availability': 0.97},
            'Stress': {'wacc_adj': 0.02, 'capex_adj': 0.20, 'revenue_adj': -0.10, 'availability': 0.94}
        }
    }
    
    results = []
    
    for scenario in ['Base', 'Risk-adjusted', 'Stress']:
        df_scenario = df.copy()
        df_scenario['scenario'] = scenario
        
        for archetype, params in risk_params.items():
            mask = df_scenario['economic_archetype'] == archetype
            if mask.sum() == 0:
                continue
                
            adj = params[scenario]
            
            # 1. Adjust WACC and recalculate annualized capex
            wacc_adj = df_scenario.loc[mask, 'wacc'] + adj['wacc_adj']
            n = df_scenario.loc[mask, 'project_life_yrs']
            pv = df_scenario.loc[mask, 'lifecycle_capex_gbp'] * (1 + adj['capex_adj'])
            
            annualized_capex = np.where(
                wacc_adj > 0,
                pv * (wacc_adj * (1 + wacc_adj)**n) / ((1 + wacc_adj)**n - 1),
                pv / n
            )
            
            # 2. Recalculate total annual cost
            total_annual_cost = annualized_capex + df_scenario.loc[mask, 'total_fom_gbp_yr']
            
            # 3. Adjust revenue (capture derate + availability)
            adjusted_revenue = (
                df_scenario.loc[mask, 'annual_gross_revenue_gbp_adj'] * 
                (1 + adj['revenue_adj']) * 
                adj['availability']
            )
            
            adjusted_mwh = (
                df_scenario.loc[mask, 'annual_mwh_delivered_adj'] * 
                adj['availability']
            )
            
            # 4. Recalculate LCOS
            lcos = np.where(
                adjusted_mwh > 0,
                total_annual_cost / adjusted_mwh,
                np.nan
            )
            
            # 5. Recalculate residual floor
            actual_rev_per_mw = adjusted_revenue / df_scenario.loc[mask, 'mw_capacity']
            required_rev_per_mw = total_annual_cost / df_scenario.loc[mask, 'mw_capacity']
            
            residual_floor = np.maximum(
                0,
                required_rev_per_mw - actual_rev_per_mw - df_scenario.loc[mask, 'total_stacked_revenue_per_mw_yr']
            )
            
            # Store results
            df_scenario.loc[mask, 'wacc_adjusted'] = wacc_adj
            df_scenario.loc[mask, 'lcos_adjusted'] = lcos
            df_scenario.loc[mask, 'annual_revenue_adjusted'] = adjusted_revenue
            df_scenario.loc[mask, 'annual_mwh_adjusted'] = adjusted_mwh
            df_scenario.loc[mask, 'residual_floor_adjusted'] = residual_floor
        
        results.append(df_scenario)
    
    df_all = pd.concat(results, ignore_index=True)
    print(f"  ✅ Risk scenarios applied (Base, Risk-adjusted, Stress).")
    return df_all

df_risk = apply_risk_scenarios(df_baseline)

# %% [markdown]
# ## 3. Correction 11: Support Justification Ratio (SJR)

# %%
def calculate_sjr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Support Justification Ratio.
    SJR = Proposed Cap-and-Floor / Residual Floor Requirement
    """
    print("\n[3.1] Calculating Support Justification Ratio (Correction 11)...")
    
    # Assume Ofgem proposes a uniform £150k/MW/yr floor for all LDES
    proposed_floor = 150_000
    df['proposed_floor_per_mw_yr'] = proposed_floor
    
    # Calculate raw SJR
    df['sjr'] = np.where(
        df['residual_floor_adjusted'] > 0,
        df['proposed_floor_per_mw_yr'] / df['residual_floor_adjusted'],
        np.nan  # If no residual floor needed, SJR is NaN (will be labeled below)
    )
    
    # Add SJR label based on the reviewer's feedback
    df['sjr_label'] = np.where(
        df['residual_floor_adjusted'] == 0,
        'No floor required',
        np.where(
            df['sjr'] > 2.0, 'Strong',
            np.where(df['sjr'] >= 1.0, 'Marginal', 'Consumer risk')
        )
    )
    
    print(f"  ✅ SJR calculated (proposed floor: £{proposed_floor/1000:.0f}k/MW/yr).")
    return df

df_risk = calculate_sjr(df_risk)

# %% [markdown]
# ## 4. Output Construction & Export

# %%
OUTPUT_COLUMNS = df_risk.columns.tolist()
df_output = df_risk[OUTPUT_COLUMNS].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_output.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")

print(f"\n[4.1] ✅ Output written → {OUTPUT_PATH}")

# %% [markdown]
# ## 5. Diagnostic Summary

# %%
print("\n" + "=" * 75)
print("NB24 — DIAGNOSTIC SUMMARY (Technology Risk Stress Test)")
print("=" * 75)

print("\n Risk Scenario Results (per archetype and scenario):")
summary = df_output.groupby(['economic_archetype', 'scenario']).agg(
    n_projects=("project_id", "count"),
    lcos=("lcos_adjusted", "mean"),
    residual_floor=("residual_floor_adjusted", "mean"),
    sjr=("sjr", "mean"),
    sjr_label=("sjr_label", "first") # 'first' works because the label is identical for the whole group
)

# Formatting
summary["lcos"] = summary["lcos"].apply(lambda x: f"£{x:,.0f}")
summary["residual_floor"] = summary["residual_floor"].apply(lambda x: f"£{x/1000:,.0f}k")
summary["sjr"] = summary["sjr"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

print(summary.rename(columns={
    "lcos": "Adj LCOS",
    "residual_floor": "Residual Floor",
    "sjr": "SJR",
    "sjr_label": "SJR Label"
}).to_string())

print("\n🚨 The CAES Viability Test:")
caes_df = df_output[df_output['economic_archetype'] == 'CAES']
for scenario in ['Base', 'Risk-adjusted', 'Stress']:
    scenario_df = caes_df[caes_df['scenario'] == scenario]
    residual = scenario_df['residual_floor_adjusted'].mean()
    sjr = scenario_df['sjr'].mean()
    label = scenario_df['sjr_label'].iloc[0]
    
    print(f"  {scenario}: Residual Floor £{residual/1000:.0f}k/MW/yr | SJR {sjr:.2f} [{label}]")

print("\n Technology Risk Ranking (Risk-adjusted scenario):")
risk_adj = df_output[df_output['scenario'] == 'Risk-adjusted'].copy()

archetype_ranking = (
    risk_adj
    .groupby('economic_archetype')
    .agg(
        n_projects=("project_id", "count"),
        mean_residual_floor=("residual_floor_adjusted", "mean"),
        min_residual_floor=("residual_floor_adjusted", "min"),
        max_residual_floor=("residual_floor_adjusted", "max"),
        mean_sjr=("sjr", "mean")
    )
    .sort_values('mean_residual_floor')
)

for tech, row in archetype_ranking.iterrows():
    print(
        f"  {tech}: avg £{row['mean_residual_floor']/1000:,.0f}k/MW/yr "
        f"(range £{row['min_residual_floor']/1000:,.0f}k–£{row['max_residual_floor']/1000:,.0f}k, "
        f"SJR {row['mean_sjr']:.2f})"
    )

print("\n" + "─" * 75)
print("NB24 complete. Output: ldes_portfolio_risk_adjusted.parquet")
print("─" * 75)
print("Next: Final synthesis for Ofgem consultation note")

# %%
print("\n📈 Technology Risk Ranking (Risk-adjusted scenario):")
risk_adj = df_output[df_output['scenario'] == 'Risk-adjusted'].copy()

archetype_ranking = (
    risk_adj
    .groupby('economic_archetype')
    .agg(
        n_projects=("project_id", "count"),
        mean_residual_floor=("residual_floor_adjusted", "mean"),
        min_residual_floor=("residual_floor_adjusted", "min"),
        max_residual_floor=("residual_floor_adjusted", "max"),
        mean_sjr=("sjr", "mean")
    )
    .sort_values('mean_residual_floor')
)

for tech, row in archetype_ranking.iterrows():
    print(
        f"{tech}: avg £{row['mean_residual_floor']/1000:,.0f}k/MW/yr "
        f"(range £{row['min_residual_floor']/1000:,.0f}k–£{row['max_residual_floor']/1000:,.0f}k, "
        f"SJR {row['mean_sjr']:.2f})"
    )

# %%
print("\n" + "=" * 75)
print("🔍 LI-ION LOCATION SPLIT — Geographic Constraint Exposure")
print("=" * 75)

li_ion_risk = df_output[
    (df_output['economic_archetype'] == 'Li-ion') & 
    (df_output['scenario'] == 'Risk-adjusted')
].copy()

# Split by RRT exposure
li_ion_constrained = li_ion_risk[li_ion_risk['rrt_applicable'] == True]
li_ion_unconstrained = li_ion_risk[li_ion_risk['rrt_applicable'] == False]

print(f"\nConstrained Portfolio (North Scotland, {len(li_ion_constrained)} projects):")
print(f"  Projects: {', '.join(li_ion_constrained['project_name'].tolist())}")
print(f"  Avg residual floor: £{li_ion_constrained['residual_floor_adjusted'].mean()/1000:.0f}k/MW/yr")
print(f"  Range: £{li_ion_constrained['residual_floor_adjusted'].min()/1000:.0f}k–£{li_ion_constrained['residual_floor_adjusted'].max()/1000:.0f}k/MW/yr")
print(f"  Avg duration: {li_ion_constrained['duration_hours'].mean():.1f}h")
print(f"  SJR: {li_ion_constrained['sjr'].mean():.2f}")

print(f"\nUnconstrained Portfolio (Central/East England, {len(li_ion_unconstrained)} projects):")
print(f"  Projects: {', '.join(li_ion_unconstrained['project_name'].tolist())}")
print(f"  Avg residual floor: £{li_ion_unconstrained['residual_floor_adjusted'].mean()/1000:.0f}k/MW/yr")
print(f"  Range: £{li_ion_unconstrained['residual_floor_adjusted'].min()/1000:.0f}k–£{li_ion_unconstrained['residual_floor_adjusted'].max()/1000:.0f}k/MW/yr")
print(f"  Avg duration: {li_ion_unconstrained['duration_hours'].mean():.1f}h")
print(f"  SJR: {li_ion_unconstrained['sjr'].mean():.2f}")

split_ratio = (li_ion_constrained['residual_floor_adjusted'].mean() / 
               li_ion_unconstrained['residual_floor_adjusted'].mean())
print(f"\n⚡ Split Ratio: {split_ratio:.2f}x")
print(f"   North Scotland Li-ion needs {split_ratio:.1f}x more support than Southern Li-ion")
print(f"   This proves Cap-and-Floor must be location-specific, not just technology-specific.")

# %%
print("="*75)
print("TRACEABILITY: 1.63x Geographic Divergence within Li-ion")
print("="*75)

# Load NB24 risk-adjusted output
df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

print(f"\nSource file: {PROCESSED_DIR / 'ldes_portfolio_risk_adjusted.parquet'}")
print(f"Relevant columns: residual_floor_adjusted, rrt_applicable, economic_archetype")

# Filter to Li-ion, Risk-adjusted scenario
li_ion_risk = df_risk[(df_risk['economic_archetype'] == 'Li-ion') & 
                       (df_risk['scenario'] == 'Risk-adjusted')]

print(f"\nLi-ion projects in risk-adjusted scenario: {len(li_ion_risk)}")

# Split by RRT exposure
constrained = li_ion_risk[li_ion_risk['rrt_applicable'] == True]
unconstrained = li_ion_risk[li_ion_risk['rrt_applicable'] == False]

print(f"\nConstrained Portfolio (rrt_applicable == True):")
print(f"  Projects: {len(constrained)}")
for _, row in constrained.iterrows():
    print(f"    {row['project_name']:25s} | Residual floor: £{row['residual_floor_adjusted']/1000:.0f}k/MW/yr")
avg_constrained = constrained['residual_floor_adjusted'].mean()
print(f"  Average: £{avg_constrained/1000:.0f}k/MW/yr")

print(f"\nUnconstrained Portfolio (rrt_applicable == False):")
print(f"  Projects: {len(unconstrained)}")
for _, row in unconstrained.iterrows():
    print(f"    {row['project_name']:25s} | Residual floor: £{row['residual_floor_adjusted']/1000:.0f}k/MW/yr")
avg_unconstrained = unconstrained['residual_floor_adjusted'].mean()
print(f"  Average: £{avg_unconstrained/1000:.0f}k/MW/yr")

split_ratio = avg_constrained / avg_unconstrained
print(f"\n✅ Geographic divergence ratio: {split_ratio:.2f}x")
print(f"   (Calculated as: mean(residual_floor_adjusted where rrt_applicable==True) / "
      f"mean(residual_floor_adjusted where rrt_applicable==False))")
print(f"   for economic_archetype == 'Li-ion' and scenario == 'Risk-adjusted'")

# %%
import numpy as np
import pandas as pd
from scipy import stats

# Load the data
df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

# Filter to Li-ion, Risk-adjusted scenario
li_ion = df_risk[(df_risk['economic_archetype'] == 'Li-ion') & 
                  (df_risk['scenario'] == 'Risk-adjusted')].copy()

print("="*75)
print("DURATION vs RESIDUAL FLOOR ANALYSIS — Li-ion Projects")
print("="*75)

print("\n📊 Raw Data (sorted by duration):")
print(f"{'Project':<25s} {'Duration':<10s} {'RRT':<6s} {'Residual Floor':<15s}")
print("-"*75)

li_ion_sorted = li_ion.sort_values('duration_hours')
for _, row in li_ion_sorted.iterrows():
    rrt_flag = "✓" if row['rrt_applicable'] else "✗"
    print(f"{row['project_name']:<25s} {row['duration_hours']:<10.1f} {rrt_flag:<6s} "
          f"£{row['residual_floor_adjusted']/1000:<14.0f}k")

# Split into constrained and unconstrained
constrained = li_ion[li_ion['rrt_applicable'] == True]
unconstrained = li_ion[li_ion['rrt_applicable'] == False]

print(f"\n📈 Regression: Residual Floor vs Duration (Unconstrained projects only)")
print("-"*75)

# Fit linear regression to unconstrained projects
x_unc = unconstrained['duration_hours'].values
y_unc = unconstrained['residual_floor_adjusted'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x_unc, y_unc)

print(f"Slope: £{slope/1000:.1f}k per hour of duration")
print(f"Intercept: £{intercept/1000:.0f}k")
print(f"R²: {r_value**2:.3f}")
print(f"p-value: {p_value:.4f}")

print(f"\nPredicted model: Residual Floor = £{intercept/1000:.0f}k + £{slope/1000:.1f}k × Duration")

# Predict what constrained projects would cost if they were unconstrained
print(f"\n🔍 Duration-Controlled Geographic Effect:")
print("-"*75)
print(f"{'Project':<25s} {'Duration':<10s} {'Actual':<12s} {'Predicted':<12s} {'Difference':<12s} {'% Diff':<8s}")
print("-"*75)

differences = []
for _, row in constrained.iterrows():
    duration = row['duration_hours']
    actual = row['residual_floor_adjusted']
    predicted = intercept + slope * duration
    diff = actual - predicted
    pct_diff = (diff / predicted) * 100
    differences.append(pct_diff)
    
    print(f"{row['project_name']:<25s} {duration:<10.1f} £{actual/1000:<11.0f}k "
          f"£{predicted/1000:<11.0f}k £{diff/1000:<+11.0f}k {pct_diff:<+7.1f}%")

avg_pct_diff = np.mean(differences)
print("-"*75)
print(f"Average geographic premium (duration-controlled): {avg_pct_diff:+.1f}%")
print(f"Range: {min(differences):+.1f}% to {max(differences):+.1f}%")

print(f"\n✅ KEY FINDING:")
print(f"   Once duration is controlled for, North Scotland Li-ion projects carry")
print(f"   a {avg_pct_diff:+.1f}% geographic premium due to RRT constraint exposure.")
print(f"   The original 1.63x ratio was confounded by duration variation.")

# %%
print("\n" + "="*75)
print("MULTIPLE REGRESSION: Isolating Duration vs Geography Effects")
print("="*75)

# Prepare data for regression
X = li_ion[['duration_hours', 'rrt_applicable']].copy()
X['rrt_applicable'] = X['rrt_applicable'].astype(int)  # Convert boolean to 0/1
y = li_ion['residual_floor_adjusted'].values

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Run OLS regression
import statsmodels.api as sm
model = sm.OLS(y, X_with_const).fit()

print(f"\nRegression Results:")
print(f"{'Variable':<20s} {'Coefficient':<15s} {'Std Error':<12s} {'p-value':<10s}")
print("-"*75)
print(f"{'Intercept':<20s} £{model.params['const']/1000:<14.0f}k "
      f"£{model.bse['const']/1000:<11.0f}k {model.pvalues['const']:<10.4f}")
print(f"{'Duration (hours)':<20s} £{model.params['duration_hours']/1000:<14.1f}k "
      f"£{model.bse['duration_hours']/1000:<11.1f}k {model.pvalues['duration_hours']:<10.4f}")
print(f"{'RRT exposure':<20s} £{model.params['rrt_applicable']/1000:<14.0f}k "
      f"£{model.bse['rrt_applicable']/1000:<11.0f}k {model.pvalues['rrt_applicable']:<10.4f}")

print(f"\nModel R²: {model.rsquared:.3f}")
print(f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")

print(f"\n📊 Interpretation:")
print(f"   • Each additional hour of duration adds £{model.params['duration_hours']/1000:.1f}k/MW/yr to residual floor")
print(f"   • RRT exposure adds £{model.params['rrt_applicable']/1000:.0f}k/MW/yr (duration-controlled)")
print(f"   • At average duration of 17.2h, RRT effect is {model.params['rrt_applicable']/1000:.0f}k / "
      f"({model.params['const']/1000:.0f}k + {model.params['duration_hours']/1000:.1f}k × 17.2) = "
      f"{model.params['rrt_applicable'] / (model.params['const'] + model.params['duration_hours'] * 17.2) * 100:.1f}%")

# %% [markdown]
# ## NB24 Finding — Technology Risk Adjustment Reveals Location-Specific Support Requirements
#
# Applying technology-specific risk adjustments to the real Window 1 portfolio reveals that Cap-and-Floor support cannot be a uniform instrument, even within a single technology archetype.
#
# The Li-ion Location Split Is the Strongest Evidence:
#
# The 11 Li-ion projects show a decisive geographic split:
#
# | **Portfolio** | **Projects** | **Avg Duration** | **Avg Residual Floor** | **SJR** |
# |---------------|--------------|-------------------|--------------------------|---------|
# | Constrained (North Scotland) | Netherton, New Deer, Rigifa, Fyrish | 17.2h | £693k/MW/yr | 0.22 |
# | Unconstrained (Central/East England) | Long Stratton, East Claydon, Ocker Hill, Sundon, Drakelow, Springwell, Thornton | 10.7h | £426k/MW/yr | 0.37 |
#
#
# This is a 1.63x difference in support requirement for the same technology. The Constrained Portfolio has longer duration (17.2h vs 10.7h), which should reduce their LCOS, but all four projects sit behind the SSEN-S constraint interface, which increases their curtailment penalty and residual floor requirement.
#
# The Net Effect: Geographic constraint exposure dominates the duration advantage. A uniform £150k/MW/yr Cap-and-Floor would:
#
# - Severely under-support the Constrained Portfolio by £543k/MW/yr (SJR 0.22)
# - Under-support the Unconstrained Portfolio by £276k/MW/yr (SJR 0.37)
#
# The PSH Contrast: PSH shows the opposite pattern. All three PSH projects (Coire Glas, Earba, Loch Kemp) sit behind the same SSEN-S constraint, but their long duration (15–32h) gives them 85% duration resilience, allowing them to time-shift discharge and mitigate curtailment. As a result, PSH's residual floor is only £60k/MW/yr (SJR 2.66), meaning a uniform floor would over-support PSH by £90k/MW/yr.
#
# The North Scotland Paradox: The same constraint zone hosts two very different support cases:
#
# - PSH (Coire Glas, Earba, Loch Kemp): £60k/MW/yr residual floor → over-supported at £150k
# - Li-ion (Field Netherton, New Deer, Rigifa, Fyrish): £693k/MW/yr residual floor → severely under-supported at £150k
#
# This is a 12x difference in support requirement for assets in the same geographic constraint zone, demonstrating that technology type and duration matter more than location alone.
#
# The Central Finding: Cap-and-Floor must be calibrated to both technology type and geographic location. The same technology (Li-ion) has a 1.63x difference in support requirement based solely on location. Different technologies (PSH vs Li-ion) in the same location have a 12x difference in support requirement. A one-size-fits-all approach would simultaneously over-support some projects and under-support others, distorting the technology mix and wasting consumer money.
#
# NB24 does not show that any technology is unviable. It shows that the level of regulatory support must be precisely calibrated to the intersection of technology lifecycle costs, duration characteristics, delivery risk, and geographic constraint exposure.

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(13, 7))

# Technologies and their residual floors (Risk-adjusted scenario)
technologies = ['PSH', 'CAES', 'Flow', 'Li-ion']
residual_floors = [60, 231, 252, 523]  # Li-ion is the average
proposed_floor = 150

# Li-ion split
li_ion_unconstrained = 426  # Central/East England (7 projects)
li_ion_constrained = 693    # North Scotland (4 projects)

# Color coding
colors = ['#2ca02c', '#d62728', '#d62728', '#d62728']  # Green for PSH (over-supported), red for others

x = np.arange(len(technologies))
width = 0.6

# Create bars
bars = ax.bar(x, residual_floors, width, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Split the Li-ion bar into two sections
li_ion_bar = bars[3]
li_ion_x = li_ion_bar.get_x()
li_ion_width = li_ion_bar.get_width()

# Remove the original Li-ion bar and replace with two sub-bars
li_ion_bar.remove()

# Create two sub-bars for Li-ion
sub_width = li_ion_width * 0.4
sub_bars = ax.bar([li_ion_x - sub_width/2 - 0.05, li_ion_x + sub_width/2 + 0.05], 
                  [li_ion_unconstrained, li_ion_constrained], 
                  sub_width, 
                  color=['#ff9896', '#d62728'], 
                  alpha=0.8, 
                  edgecolor='black', 
                  linewidth=1.5)

# Add the proposed floor line
ax.axhline(y=proposed_floor, color='blue', linestyle='--', linewidth=2.5, 
           label=f'Proposed £{proposed_floor}k/MW/yr Floor', zorder=10)

# Add value labels on bars
for i, (bar, residual) in enumerate(zip(bars[:3], residual_floors[:3])):
    height = bar.get_height()
    gap = residual - proposed_floor
    
    # Show residual floor value
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'£{residual}k',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Show gap annotation
    if gap < 0:
        ax.text(bar.get_x() + bar.get_width()/2., height - 30,
                f'£{abs(gap)}k OVER',
                ha='center', va='top', fontsize=10, fontweight='bold', color='#2ca02c')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height + 40,
                f'£{gap}k SHORT',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d62728')

# Add labels for Li-ion sub-bars
for i, (sub_bar, value, label) in enumerate(zip(sub_bars, 
                                                  [li_ion_unconstrained, li_ion_constrained],
                                                  ['Unconstrained\n(Central/East)', 'Constrained\n(North Scotland)'])):
    height = sub_bar.get_height()
    gap = value - proposed_floor
    
    # Show value
    ax.text(sub_bar.get_x() + sub_bar.get_width()/2., height + 10,
            f'£{value}k',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Show gap
    ax.text(sub_bar.get_x() + sub_bar.get_width()/2., height + 40,
            f'£{gap}k SHORT',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d62728')
    
    # Add sub-bar labels below
    ax.text(sub_bar.get_x() + sub_bar.get_width()/2., 20,
            label,
            ha='center', va='bottom', fontsize=9, fontstyle='italic')

# Add bracket showing the Li-ion split
ax.annotate('', xy=(li_ion_x - 0.35, li_ion_constrained + 70), 
            xytext=(li_ion_x + 0.35, li_ion_constrained + 70),
            arrowprops=dict(arrowstyle='-', color='black', lw=2))
ax.text(li_ion_x, li_ion_constrained + 85, '1.63× location split', 
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# Formatting
ax.set_ylabel('Residual Floor Requirement (£k/MW/yr)', fontsize=13, fontweight='bold')
ax.set_title('Technology-Specific Support Requirements: One Floor/One Location Does Not Fit All', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(technologies, fontsize=12, fontweight='bold')
ax.set_ylim(0, 850)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.legend(fontsize=11, loc='upper left')

# Add policy summary text box
summary_text = ('Policy Implications:\n'
                '• PSH: Over-supported by £90k → consumer cost\n'
                '• CAES: Under-supported by £81k → project unviable\n'
                '• Flow: Under-supported by £102k → project unviable\n'
                '• Li-ion: Under-supported by £276k–£543k\n'
                '  (varies 1.63× by location)\n\n'
                'Conclusion: Cap-and-Floor must be calibrated to both\n'
                'technology type AND geographic location')

ax.text(0.58, 0.88, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.show()

# %% [markdown]
# Check

# %%
print("🔍 NB24 Input Check:")
df_nb24 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_stacked.parquet")
print(f"  Portfolio loaded: {len(df_nb24)} projects")
print(f"  Has stacked revenue columns: {'residual_floor_per_mw_yr' in df_nb24.columns}")
print(f"  Has cm_revenue_per_mw_yr: {'cm_revenue_per_mw_yr' in df_nb24.columns}")
print(f"  Has bm_revenue_per_mw_yr: {'bm_revenue_per_mw_yr' in df_nb24.columns}")

print("\n🔍 NB24 Risk Scenario Check:")
df_nb24_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

for scenario in ['Base', 'Risk-adjusted', 'Stress']:
    scenario_data = df_nb24_risk[df_nb24_risk['scenario'] == scenario]
    print(f"\n  {scenario}:")
    
    for archetype in scenario_data['economic_archetype'].unique():
        arch_data = scenario_data[scenario_data['economic_archetype'] == archetype]
        
        # Calculate aggregate residual (mean of residuals)
        avg_residual = arch_data['residual_floor_adjusted'].mean()
        
        # Calculate aggregate SJR from average residual (NOT average of SJRs)
        proposed_floor = 150_000
        if avg_residual > 0:
            aggregate_sjr = proposed_floor / avg_residual
        else:
            aggregate_sjr = np.nan
        
        print(f"    {archetype:10s} | Residual: £{avg_residual/1000:.0f}k | "
              f"Aggregate SJR: {aggregate_sjr:.2f}")

print("\n🔍 NB24 Li-ion Location Split Check:")
li_ion_risk = df_nb24_risk[(df_nb24_risk['economic_archetype'] == 'Li-ion') & (df_nb24_risk['scenario'] == 'Risk-adjusted')]
constrained = li_ion_risk[li_ion_risk['rrt_applicable'] == True]
unconstrained = li_ion_risk[li_ion_risk['rrt_applicable'] == False]
print(f"  Constrained (North Scotland): {len(constrained)} projects, avg £{constrained['residual_floor_adjusted'].mean()/1000:.0f}k/MW/yr")
print(f"  Unconstrained (Central/East): {len(unconstrained)} projects, avg £{unconstrained['residual_floor_adjusted'].mean()/1000:.0f}k/MW/yr")
split_ratio = constrained['residual_floor_adjusted'].mean() / unconstrained['residual_floor_adjusted'].mean()
print(f"  Split ratio: {split_ratio:.2f}x")

print("\n🔍 NB24 Output Check:")
output_path = PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet"
print(f"  Output exists: {output_path.exists()}")
if output_path.exists():
    df_check = pd.read_parquet(output_path)
    print(f"  Rows: {len(df_check)}")
    print(f"  Has residual_floor_adjusted: {'residual_floor_adjusted' in df_check.columns}")
    print(f"  Has sjr: {'sjr' in df_check.columns}")

# %%
print("\n🔍 Base Scenario PSH Deep Dive:")
df_base = df_nb24_risk[df_nb24_risk['scenario'] == 'Base']
psh_base = df_base[df_base['economic_archetype'] == 'PSH']

for _, row in psh_base.iterrows():
    print(f"  {row['project_name']:20s} | Residual: £{row['residual_floor_adjusted']/1000:.0f}k | "
          f"Proposed Floor: £{row['proposed_floor_per_mw_yr']/1000:.0f}k | SJR: {row['sjr']:.2f}")

print(f"\n  Proposed floor values: {psh_base['proposed_floor_per_mw_yr'].unique()}")
print(f"  Residual floor values: {psh_base['residual_floor_adjusted'].unique()}")

# %%
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("/home/ndrew/NW_bess_constraints/data/processed")

print("="*75)
print("AUDIT 1: CHECKING FOR DOUBLE-COUNTING")
print("="*75)

# Load both outputs
df_nb21 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_revenues.parquet")
df_nb22 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_curtailment.parquet")

# Get Field Netherton
netherton_nb21 = df_nb21[df_nb21['project_name'] == 'Field Netherton'].iloc[0]
netherton_nb22 = df_nb22[df_nb22['project_name'] == 'Field Netherton'].iloc[0]

print("\n📊 NB21 (Market Revenue):")
print(f"  Annual MWh delivered: {netherton_nb21['annual_mwh_delivered']:,.0f}")
print(f"  Annual gross revenue: £{netherton_nb21['annual_gross_revenue_gbp']:,.0f}")
print(f"  LCOS: £{netherton_nb21['lcos_gbp_mwh']:.2f}")

print("\n📊 NB22 (Curtailment Applied):")
print(f"  Annual MWh delivered (adj): {netherton_nb22['annual_mwh_delivered_adj']:,.0f}")
print(f"  Annual gross revenue (adj): £{netherton_nb22['annual_gross_revenue_gbp_adj']:,.0f}")
print(f"  LCOS (adj): £{netherton_nb22['lcos_gbp_mwh_adj']:.2f}")

print("\n🔍 REDUCTIONS:")
mwh_reduction = (1 - netherton_nb22['annual_mwh_delivered_adj']/netherton_nb21['annual_mwh_delivered'])*100
rev_reduction = (1 - netherton_nb22['annual_gross_revenue_gbp_adj']/netherton_nb21['annual_gross_revenue_gbp'])*100
lcos_increase = (netherton_nb22['lcos_gbp_mwh_adj']/netherton_nb21['lcos_gbp_mwh'] - 1)*100

print(f"  MWh reduction: {mwh_reduction:.1f}%")
print(f"  Revenue reduction: {rev_reduction:.1f}%")
print(f"  LCOS increase: {lcos_increase:.1f}%")

print("\n🔍 CHECKING NB21 NARROW PIPE PENALTY:")
print(f"  Columns with 'penalty' or 'narrow': {[c for c in df_nb21.columns if 'penalty' in c.lower() or 'narrow' in c.lower()]}")
print(f"  NB21 discharge price: £{netherton_nb21['avg_discharge_price_gbp']:.2f}")

print("\n🔍 CHECKING NB22 CORRECTION:")
print(f"  Base curtailment factor: {netherton_nb22['base_curtailment_factor']:.3f}")
print(f"  Duration resilience factor: {netherton_nb22['duration_resilience_factor']:.3f}")
print(f"  Effective curtailment factor: {netherton_nb22['effective_curtailment_factor']:.3f}")

# %%
print("="*75)
print("AUDIT 2: FINDING DURATION RESILIENCE FORMULA")
print("="*75)

# Check all RRT projects
rrt_projects = df_nb22[df_nb22['rrt_applicable'] == True].copy()

print("\n📊 Duration vs Resilience Factor:")
print(f"{'Project':<25s} {'Duration':<10s} {'Resilience':<12s} {'Ratio':<10s}")
print("-"*60)

for _, row in rrt_projects.sort_values('duration_hours').iterrows():
    duration = row['duration_hours']
    resilience = row['duration_resilience_factor']
    ratio = resilience / (duration / 32)  # Test if it's duration/32
    print(f"{row['project_name']:<25s} {duration:<10.1f} {resilience:<12.3f} {ratio:<10.3f}")

print("\n🔍 Testing formulas:")
print("  If ratio ≈ 1.0 for all, then: resilience = duration / 32")
print("  Otherwise, check NB22 code for the actual formula")

# %%
print("="*75)
print("AUDIT 3: RERUNNING REGRESSION ON CURRENT DATA")
print("="*75)

from scipy import stats
import statsmodels.api as sm

# Load NB24 risk-adjusted data
df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

# Filter to Li-ion, Risk-adjusted
li_ion = df_risk[(df_risk['economic_archetype'] == 'Li-ion') & 
                  (df_risk['scenario'] == 'Risk-adjusted')].copy()

print("\n📊 Current Li-ion data (sorted by duration):")
print(f"{'Project':<25s} {'Duration':<10s} {'RRT':<6s} {'Residual':<12s}")
print("-"*60)

for _, row in li_ion.sort_values('duration_hours').iterrows():
    rrt = "✓" if row['rrt_applicable'] else "✗"
    print(f"{row['project_name']:<25s} {row['duration_hours']:<10.1f} {rrt:<6s} "
          f"£{row['residual_floor_adjusted']/1000:<11.0f}k")

# Run regression
X = li_ion[['duration_hours', 'rrt_applicable']].copy()
X['rrt_applicable'] = X['rrt_applicable'].astype(int)
X = sm.add_constant(X)
y = li_ion['residual_floor_adjusted'].values

model = sm.OLS(y, X).fit()

print("\n📈 Regression Results:")
print(f"  Intercept: £{model.params['const']/1000:.0f}k (p={model.pvalues['const']:.4f})")
print(f"  Duration: £{model.params['duration_hours']/1000:.1f}k per hour (p={model.pvalues['duration_hours']:.4f})")
print(f"  RRT: £{model.params['rrt_applicable']/1000:.0f}k (p={model.pvalues['rrt_applicable']:.4f})")
print(f"  R²: {model.rsquared:.3f}")

# Check individual premiums
constrained = li_ion[li_ion['rrt_applicable'] == True]
print("\n🔍 Duration-controlled premiums:")
for _, row in constrained.iterrows():
    predicted = model.params['const'] + model.params['duration_hours'] * row['duration_hours']
    actual = row['residual_floor_adjusted']
    premium = actual - predicted
    pct_premium = (premium / predicted) * 100
    print(f"  {row['project_name']:<25s} {row['duration_hours']:<5.1f}h | "
          f"Predicted: £{predicted/1000:.0f}k | Actual: £{actual/1000:.0f}k | "
          f"Premium: {pct_premium:+.1f}%")

# %%
print("="*75)
print("AUDIT 4: DOES NB22 BUG AFFECT PSH PROJECTS?")
print("="*75)

# Load NB21 and NB22 outputs
df_nb21 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_revenues.parquet")
df_nb22 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_curtailment.parquet")

# Check all three PSH projects
psh_projects = ['Coire Glas', 'Earba', 'Loch Kemp Storage']

print("\n📊 PSH Projects Through NB22 Curtailment:")
print(f"{'Project':<25s} {'Duration':<10s} {'Resilience':<12s} {'Effective':<12s} {'MWh Before':<15s} {'MWh After':<15s} {'Reduction':<12s}")
print("-"*100)

for project in psh_projects:
    nb21_row = df_nb21[df_nb21['project_name'] == project].iloc[0]
    nb22_row = df_nb22[df_nb22['project_name'] == project].iloc[0]
    
    duration = nb22_row['duration_hours']
    resilience = nb22_row['duration_resilience_factor']
    effective = nb22_row['effective_curtailment_factor']
    
    mwh_before = nb21_row['annual_mwh_delivered']
    mwh_after = nb22_row['annual_mwh_delivered_adj']
    reduction_pct = (1 - mwh_after / mwh_before) * 100
    
    print(f"{project:<25s} {duration:<10.1f} {resilience:<12.3f} {effective:<12.3f} "
          f"{mwh_before:<15,.0f} {mwh_after:<15,.0f} {reduction_pct:<11.1f}%")

print("\n🔍 Expected vs Actual Reductions:")
print("  • Coire Glas (32h, resilience=1.0): Expected 0% reduction (effective=1.0)")
print("  • Loch Kemp (22h, resilience=0.929): Expected 5.8% reduction (effective=0.942)")
print("  • Earba (15h, resilience=0.625): Expected 30.7% reduction (effective=0.693)")

print("\n🔍 Checking LCOS Changes:")
for project in psh_projects:
    nb21_row = df_nb21[df_nb21['project_name'] == project].iloc[0]
    nb22_row = df_nb22[df_nb22['project_name'] == project].iloc[0]
    
    lcos_before = nb21_row['lcos_gbp_mwh']
    lcos_after = nb22_row['lcos_gbp_mwh_adj']
    effective = nb22_row['effective_curtailment_factor']
    
    expected_lcos_after = lcos_before / effective
    actual_change = (lcos_after / lcos_before - 1) * 100
    expected_change = (expected_lcos_after / lcos_before - 1) * 100
    
    print(f"\n  {project}:")
    print(f"    LCOS before: £{lcos_before:.2f}")
    print(f"    LCOS after (actual): £{lcos_after:.2f} ({actual_change:+.1f}%)")
    print(f"    LCOS after (expected): £{expected_lcos_after:.2f} ({expected_change:+.1f}%)")
    print(f"    Match: {'✅ YES' if abs(actual_change - expected_change) < 0.1 else '❌ NO - BUG DETECTED'}")

# %%
print("\n" + "="*75)
print("AUDIT 5: PSH CAPEX SENSITIVITY SWEEP")
print("="*75)

# Load current economics data
df_econ = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_economics.parquet")

# Get PSH projects
psh = df_econ[df_econ['technology_type'] == 'PSH'].copy()

print("\n📊 Current PSH Capex Assumptions:")
print(f"  capex_mw_gbp: £1,200,000/MW")
print(f"  capex_mwh_gbp: £20,000/MWh (current)")
print(f"  lifecycle_multiplier: 1.15")

# Test different capex_mwh values
capex_mwh_values = [15000, 20000, 25000, 30000]

print("\n📈 Sensitivity Analysis:")
print(f"{'capex_mwh':<12s} {'Coire Glas':<15s} {'Earba':<15s} {'Loch Kemp':<15s} {'Avg SJR':<10s} {'Status':<15s}")
print("-"*85)

for capex_mwh in capex_mwh_values:
    sjr_values = []
    
    for _, row in psh.iterrows():
        # Recalculate lifecycle capex
        mw = row['mw_capacity']
        mwh = row['mwh_capacity']
        
        initial_capex = (mw * 1_200_000) + (mwh * capex_mwh)
        lifecycle_capex = initial_capex * 1.15
        
        # Get current residual floor (from NB24 output)
        df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")
        risk_row = df_risk[(df_risk['project_name'] == row['project_name']) & 
                           (df_risk['scenario'] == 'Risk-adjusted')].iloc[0]
        
        # Recalculate residual floor (simplified - assumes same revenue structure)
        # This is approximate; full recalculation would require rerunning NB23-NB24
        current_residual = risk_row['residual_floor_adjusted']
        current_lifecycle = row['lifecycle_capex_gbp']
        
        # Scale residual floor proportionally to lifecycle capex change
        new_residual = current_residual * (lifecycle_capex / current_lifecycle)
        
        # Calculate SJR (assuming £150k proposed floor)
        sjr = 150_000 / new_residual
        sjr_values.append(sjr)
    
    avg_sjr = np.mean(sjr_values)
    status = "Over-supported" if avg_sjr > 2.0 else "Marginal" if avg_sjr > 1.0 else "Under-supported"
    
    # Calculate individual project SJRs for display
    cg_sjr = sjr_values[0]
    earba_sjr = sjr_values[1]
    lk_sjr = sjr_values[2]
    
    print(f"£{capex_mwh/1000:.0f}k/MWh    £{150_000/cg_sjr/1000:.0f}k residual   "
          f"£{150_000/earba_sjr/1000:.0f}k residual   £{150_000/lk_sjr/1000:.0f}k residual   "
          f"{avg_sjr:<10.2f} {status:<15s}")

print("\n🔍 External Validation:")
print("  • Snowy 2.0: ~£23,000/MWh (March 2025 dollars)")
print("  • Published range: 140% to 4,300% of Snowy 2.0 benchmark")
print("  • Your £20k/MWh sits at the low, well-engineered end")
print("  • Coire Glas 2014 estimate: ~$42,000/MWh blended (power+energy)")

print("\n✅ KEY FINDING:")
print("  If SJR stays above 2.0 across £15k-£30k/MWh range, 'over-supported' is robust")
print("  If SJR drops below 2.0 within this range, finding becomes 'possibly over-supported'")

# %%
print("="*75)
print("PSH INTEREST DURING CONSTRUCTION (IDC) SENSITIVITY TEST")
print("="*75)

import numpy as np

# Load PSH data
df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")
psh_risk = df_risk[(df_risk['economic_archetype'] == 'PSH') & 
                    (df_risk['scenario'] == 'Risk-adjusted')]

print("\n📊 Current PSH Economics (No IDC):")
for _, row in psh_risk.iterrows():
    print(f"  {row['project_name']:<20s} | Residual: £{row['residual_floor_adjusted']/1000:.0f}k | "
          f"Duration: {row['duration_hours']:.0f}h")

# Test IDC impact
wacc = 0.08  # 8% WACC
construction_years = 8

print(f"\n🔍 Testing with IDC (WACC={wacc*100:.0f}%, Construction={construction_years} years):")
print(f"  Assumption: Capex accrues financing cost for {construction_years} years before revenue starts")
print(f"  Formula: effective_capex = capex × (1 + WACC)^construction_years")

idc_multiplier = (1 + wacc) ** construction_years
print(f"  IDC Multiplier: {idc_multiplier:.2f}x")

print(f"\n📊 PSH Economics WITH IDC:")
print(f"{'Project':<20s} {'Current Residual':<20s} {'With IDC':<20s} {'Change':<15s}")
print("-"*75)

for _, row in psh_risk.iterrows():
    current_residual = row['residual_floor_adjusted']
    
    # IDC increases the effective capex burden, which increases residual floor
    # Simplified calculation: scale residual proportionally to IDC impact
    # In reality, this would require rerunning NB20-NB24 with IDC
    idc_residual = current_residual * idc_multiplier
    
    change = idc_residual - current_residual
    print(f"{row['project_name']:<20s} £{current_residual/1000:<19.0f}k "
          f"£{idc_residual/1000:<19.0f}k £{change/1000:<+14.0f}k")

print("\n⚠️  IMPORTANT CAVEAT:")
print("  This is a simplified calculation. Full IDC analysis requires:")
print("  • Year-by-year construction cash flow modeling")
print("  • Proper debt/equity structure (not just WACC)")
print("  • Revenue start date alignment")
print("  • Tax and depreciation treatment")
print("\n  However, this test shows whether the zero-floor finding is")
print("  likely to survive a proper IDC calculation.")

print("\n✅ RECOMMENDED LANGUAGE:")
print("  'Under central scenario assumptions, PSH requires minimal incremental")
print("  floor support relative to Li-ion, with residual floors at or near zero")
print("  before adjusting for construction-period financing risk. PSH's 7-10 year")
print("  construction period creates significant Interest During Construction (IDC)")
print("  that is not captured in the current model.'")

# %%
print("="*75)
print("PSH INTEREST DURING CONSTRUCTION (IDC) — STRESS TEST")
print("="*75)

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("/home/ndrew/NW_bess_constraints/data/processed")

# Load the risk-adjusted data (has everything we need)
df_nb24 = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

# Get PSH projects
psh = df_nb24[(df_nb24['economic_archetype'] == 'PSH') & 
              (df_nb24['scenario'] == 'Risk-adjusted')].copy()

print("\n📊 Current PSH Economics (No IDC):")
print(f"{'Project':<25s} {'Duration':<10s} {'Residual Floor':<20s}")
print("-"*60)

for _, row in psh.iterrows():
    print(f"{row['project_name']:<25s} {row['duration_hours']:<10.1f} "
          f"£{row['residual_floor_adjusted']/1000:<19.0f}k")

# IDC Calculation
wacc = 0.08  # 8% WACC
construction_years = 8

print(f"\n🔍 IDC Calculation Parameters:")
print(f"  WACC: {wacc*100:.0f}%")
print(f"  Construction period: {construction_years} years")
print(f"  Assumption: Capex is spent evenly over construction period")
print(f"  Formula: IDC Factor = (1 + WACC)^(construction/2)")
print(f"           = (1 + {wacc})^({construction_years}/2)")

# Calculate IDC factor
idc_factor = (1 + wacc) ** (construction_years / 2)
print(f"\n  IDC Factor: {idc_factor:.3f}x")
print(f"  This means effective capex increases by {idc_factor:.2f}x")

print(f"\n📊 PSH Residual Floors WITH IDC:")
print(f"{'Project':<25s} {'Current':<15s} {'With IDC':<15s} {'Change':<15s}")
print("-"*75)

for _, row in psh.iterrows():
    current_residual = row['residual_floor_adjusted']
    
    # IDC increases the required revenue (and thus residual floor) by the IDC factor
    # If current residual is £0, we need to calculate what it would be with IDC
    # Simplified: required_floor_with_idc = required_floor × idc_factor
    # Then: residual_with_idc = required_floor_with_idc - actual_revenue
    
    # Get the actual revenue and required floor from the data
    actual_revenue = row['actual_revenue_per_mw_yr_adj']
    required_floor = row['required_floor_per_mw_yr_adj']
    
    # Apply IDC to required floor
    required_floor_with_idc = required_floor * idc_factor
    
    # Recalculate residual
    residual_with_idc = max(0, required_floor_with_idc - actual_revenue)
    
    change = residual_with_idc - current_residual
    
    print(f"{row['project_name']:<25s} £{current_residual/1000:<14.0f}k "
          f"£{residual_with_idc/1000:<14.0f}k £{change/1000:<+14.0f}k")

print("\n✅ KEY FINDING:")
print("  If all PSH projects still show £0 residual with IDC, the finding is robust.")
print("  If any project shows positive residual, the finding becomes conditional.")

# %%
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("/home/ndrew/NW_bess_constraints/data/processed")

print("="*75)
print("GENERATING ANNEX C: FULL PROJECT-LEVEL RESULTS")
print("="*75)

# Load the parquet file
df = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")

# Filter to Risk-adjusted scenario
df_risk = df[df['scenario'] == 'Risk-adjusted'].copy()

print(f"\n✅ Loaded {len(df_risk)} projects (Risk-adjusted scenario)")

# Build export dataframe with correct column selection
export_df = df_risk[[
    'project_name',
    'economic_archetype',
    'duration_hours',
    'mw_capacity',
    'mwh_capacity',
    'rrt_applicable',
    'grid_region',
    'initial_capex_gbp',
    'lifecycle_capex_gbp',
    'lcos_gbp_mwh_adj',
    'annual_mwh_delivered_adj',
    'annual_gross_revenue_gbp_adj',
    'cm_revenue_per_mw_yr',
    'residual_floor_adjusted'
]].copy()

# Rename columns for cleaner presentation
export_df = export_df.rename(columns={
    'project_name': 'Project',
    'economic_archetype': 'Technology',
    'duration_hours': 'Duration_h',
    'mw_capacity': 'MW_Capacity',
    'mwh_capacity': 'MWh_Capacity',
    'rrt_applicable': 'RRT_Exposed',
    'grid_region': 'Location',
    'initial_capex_gbp': 'Initial_Capex_£bn',
    'lifecycle_capex_gbp': 'Lifecycle_Capex_£bn',
    'lcos_gbp_mwh_adj': 'LCOS_£_MWh',
    'annual_mwh_delivered_adj': 'Annual_MWh_Delivered',
    'annual_gross_revenue_gbp_adj': 'Annual_Gross_Revenue_£m',
    'cm_revenue_per_mw_yr': 'CM_Revenue_£k_MW',
    'residual_floor_adjusted': 'Residual_Floor_£k_MWyr'
})

# Scale units
export_df['Initial_Capex_£bn'] = export_df['Initial_Capex_£bn'] / 1e9
export_df['Lifecycle_Capex_£bn'] = export_df['Lifecycle_Capex_£bn'] / 1e9
export_df['Annual_Gross_Revenue_£m'] = export_df['Annual_Gross_Revenue_£m'] / 1e6
export_df['CM_Revenue_£k_MW'] = export_df['CM_Revenue_£k_MW'] / 1000
export_df['Residual_Floor_£k_MWyr'] = export_df['Residual_Floor_£k_MWyr'] / 1000

# Convert RRT to Yes/No
export_df['RRT_Exposed'] = export_df['RRT_Exposed'].map({True: 'Yes', False: 'No'})

# Round numeric columns
export_df['Duration_h'] = export_df['Duration_h'].round(1)
export_df['MW_Capacity'] = export_df['MW_Capacity'].round(0).astype(int)
export_df['MWh_Capacity'] = export_df['MWh_Capacity'].round(0).astype(int)
export_df['Initial_Capex_£bn'] = export_df['Initial_Capex_£bn'].round(3)
export_df['Lifecycle_Capex_£bn'] = export_df['Lifecycle_Capex_£bn'].round(3)
export_df['LCOS_£_MWh'] = export_df['LCOS_£_MWh'].round(1)
export_df['Annual_MWh_Delivered'] = export_df['Annual_MWh_Delivered'].round(0).astype(int)
export_df['Annual_Gross_Revenue_£m'] = export_df['Annual_Gross_Revenue_£m'].round(2)
export_df['CM_Revenue_£k_MW'] = export_df['CM_Revenue_£k_MW'].round(0).astype(int)
export_df['Residual_Floor_£k_MWyr'] = export_df['Residual_Floor_£k_MWyr'].round(0).astype(int)

# Sort by technology then duration
export_df = export_df.sort_values(['Technology', 'Duration_h'])

print("\n" + "="*120)
print("ANNEX C TABLE (COPY-PASTE READY FOR MARKDOWN)")
print("="*120)
print(export_df.to_markdown(index=False))

# Save to CSV
output_path = PROCESSED_DIR / "annex_c_full_results.csv"
export_df.to_csv(output_path, index=False)
print(f"\n✅ CSV saved to: {output_path}")

# Summary statistics
print("\n" + "="*75)
print("SUMMARY STATISTICS")
print("="*75)
print(f"Total projects: {len(export_df)}")
print(f"PSH projects: {len(export_df[export_df['Technology'] == 'PSH'])}")
print(f"Li-ion projects: {len(export_df[export_df['Technology'] == 'Li-ion'])}")
print(f"CAES projects: {len(export_df[export_df['Technology'] == 'CAES'])}")
print(f"Flow projects: {len(export_df[export_df['Technology'] == 'Flow'])}")
print(f"RRT-exposed projects: {len(export_df[export_df['RRT_Exposed'] == 'Yes'])}")
print(f"Avg residual floor (all): £{export_df['Residual_Floor_£k_MWyr'].mean():.0f}k/MW/yr")
print(f"Avg residual floor (Li-ion): £{export_df[export_df['Technology'] == 'Li-ion']['Residual_Floor_£k_MWyr'].mean():.0f}k/MW/yr")
print(f"Avg residual floor (PSH): £{export_df[export_df['Technology'] == 'PSH']['Residual_Floor_£k_MWyr'].mean():.0f}k/MW/yr")

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("/home/ndrew/NW_bess_constraints/data/processed")
df_risk = pd.read_parquet(PROCESSED_DIR / "ldes_portfolio_risk_adjusted.parquet")
li_ion_risk = df_risk[(df_risk['economic_archetype'] == 'Li-ion') & 
                       (df_risk['scenario'] == 'Risk-adjusted')].copy()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 6))

# Chart 1: Duration Coefficient (Scatter + Regression)
ax1 = plt.subplot(1, 3, 1)
colors = ['#d62728' if rrt else '#1f77b4' for rrt in li_ion_risk['rrt_applicable']]
ax1.scatter(li_ion_risk['duration_hours'], li_ion_risk['residual_floor_adjusted']/1000, 
           c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)

# Add regression line
x_line = np.linspace(8, 18, 100)
y_line = -51 + 27.8 * x_line
ax1.plot(x_line, y_line, 'k--', linewidth=2, label='Model: £27.8k/hour', zorder=2)

ax1.set_xlabel('Duration (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Modelled Residual Support\n(£k/MW/yr)', fontsize=11, fontweight='bold')
ax1.set_title('Finding 1: Duration Coefficient', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(7, 19)
ax1.set_ylim(0, 650)

# Add annotation
ax1.annotate('£27.8k per hour\n(R² = 0.999)', xy=(14, 550), fontsize=10, 
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

# Chart 2: Geographic Premium (Bar Chart)
ax2 = plt.subplot(1, 3, 2)
constrained = li_ion_risk[li_ion_risk['rrt_applicable'] == True]
unconstrained = li_ion_risk[li_ion_risk['rrt_applicable'] == False]

# Calculate averages
avg_constrained = constrained['residual_floor_adjusted'].mean() / 1000
avg_unconstrained = unconstrained['residual_floor_adjusted'].mean() / 1000

bars = ax2.bar(['Unconstrained\n(Central/East)', 'Constrained\n(North Scotland)'], 
              [avg_unconstrained, avg_constrained], 
              color=['#1f77b4', '#d62728'], alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'£{height:.0f}k', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add percentage annotation
ax2.annotate('+30% premium', xy=(1, avg_constrained), xytext=(0.5, avg_constrained + 50),
            fontsize=12, fontweight='bold', color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

ax2.set_ylabel('Modelled Residual Support\n(£k/MW/yr)', fontsize=11, fontweight='bold')
ax2.set_title('Finding 2: Geographic Premium', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 700)
ax2.grid(axis='y', alpha=0.3)

# Chart 3: PSH Divergence (Comparison)
ax3 = plt.subplot(1, 3, 3)
psh_risk = df_risk[(df_risk['economic_archetype'] == 'PSH') & 
                    (df_risk['scenario'] == 'Risk-adjusted')]

avg_li_ion = li_ion_risk['residual_floor_adjusted'].mean() / 1000
avg_psh = psh_risk['residual_floor_adjusted'].mean() / 1000

bars = ax3.bar(['Li-ion\n(11 projects)', 'PSH\n(3 projects)'], 
              [avg_li_ion, avg_psh], 
              color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'£{height:.0f}k', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax3.text(bar.get_x() + bar.get_width()/2., 20,
                f'£0k', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add annotation
ax3.annotate('PSH: Near-zero\nresidual support', xy=(1, avg_psh), xytext=(0.5, 300),
            fontsize=11, fontweight='bold', color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax3.set_ylabel('Modelled Residual Support\n(£k/MW/yr)', fontsize=11, fontweight='bold')
ax3.set_title('Finding 3: PSH Divergence', fontsize=13, fontweight='bold', pad=10)
ax3.set_ylim(0, 650)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PROCESSED_DIR / 'findings_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Chart saved to:", PROCESSED_DIR / 'findings_visualization.png')
print("\nThis single figure tells the complete story:")
print("  • Left: Duration drives residual support (£27.8k/hour)")
print("  • Middle: Geography adds 30% premium (constrained vs unconstrained)")
print("  • Right: PSH needs near-zero support vs Li-ion's substantial requirements")

# %%
