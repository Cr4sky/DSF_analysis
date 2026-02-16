import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Add these constants at the top or inside main() ---
T_MIN = 30  # Don't look for peaks below 30°C
T_MAX = 90  # Don't look for peaks above 90°C

def calculate_individual_tms(df, temp_col, der_col, well_col):
    results = []
    unique_wells = df[well_col].unique()
    
    for well in unique_wells:
        well_data = df[df[well_col] == well].copy()
        
        # Convert to numeric
        well_data[temp_col] = pd.to_numeric(well_data[temp_col], errors='coerce')
        well_data[der_col] = pd.to_numeric(well_data[der_col], errors='coerce')
        
        # 1. IMPOSE LIMITS: Filter data to relevant temperature range before finding peak
        mask = (well_data[temp_col] >= T_MIN) & (well_data[temp_col] <= T_MAX)
        filtered_data = well_data[mask].dropna(subset=[temp_col, der_col])

        if filtered_data.empty:
            continue
            
        temps = filtered_data[temp_col].values
        # DSF convention is the peak of the negative derivative
        neg_der = -filtered_data[der_col].values
        
        # 2. Find peak within the limited window
        peak_idx = np.argmax(neg_der)
        tm = temps[peak_idx]
        
        # Optional: Filter out 'flat' peaks that are just noise
        if neg_der[peak_idx] < 0.05: # Adjust this threshold based on your signal strength
            continue

        results.append({well_col: well, 'Tm': tm})
        
    return pd.DataFrame(results)

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical Analysis for DSF Results")
    parser.add_argument("-f", "--file", required=True, help="Path to Data Excel file")
    parser.add_argument("-l", "--layout", required=True, help="Path to Layout Excel file")
    parser.add_argument("-s", "--sheet", default="Melt Curve Raw Data", help="Data sheet name")
    parser.add_argument("--skip", type=int, default=45, help="Rows to skip in data file")
    parser.add_argument("--temp_col", default="Temperature", help="Temperature column")
    parser.add_argument("--der_col", default="Derivative", help="Derivative column")
    parser.add_argument("--well_col", default="Well Position", help="Well identifier column")
    parser.add_argument("--control", help="Name of the control group to compare against")
    return parser.parse_args()

def calculate_individual_tms(df, temp_col, der_col, well_col):
    """Finds the peak (Tm) for every single well in the dataset."""
    results = []
    unique_wells = df[well_col].unique()
    
    for well in unique_wells:
        well_data = df[df[well_col] == well].copy()
        # Convert to numeric and flip to negative derivative
        temps = pd.to_numeric(well_data[temp_col], errors='coerce').values
        neg_der = -pd.to_numeric(well_data[der_col], errors='coerce').values
        
        # Clean data
        mask = ~np.isnan(temps) & ~np.isnan(neg_der)
        if not any(mask): continue
        
        # Find peak
        peak_idx = np.argmax(neg_der[mask])
        tm = temps[mask][peak_idx]
        results.append({well_col: well, 'Tm': tm})
        
    return pd.DataFrame(results)

def main():
    args = parse_args()

    try:
        # 1. Load Files
        data_df = pd.read_excel(args.file, sheet_name=args.sheet, skiprows=args.skip, engine='openpyxl')
        layout_df = pd.read_excel(args.layout, engine='openpyxl')
        
        data_df.columns = [str(c).strip() for c in data_df.columns]
        layout_df.columns = [str(c).strip() for c in layout_df.columns]

        # 2. Calculate Tm for every single well
        print("Calculating melting temperatures for all wells...")
        well_tms = calculate_individual_tms(data_df, args.temp_col, args.der_col, args.well_col)

        # 3. Merge with Layout to get Sample Groups
        merged_df = well_tms.merge(layout_df, on=args.well_col)
        merged_df['Sample'] = merged_df['Sample'].astype(str)

        # 4. Descriptive Statistics
        stats_summary = merged_df.groupby('Sample')['Tm'].agg(['mean', 'std', 'count', 'sem']).reset_index()
        print("\n--- Summary Statistics ---")
        print(stats_summary.to_string(index=False))

        # 5. One-Way ANOVA
        # Checks if there is any difference at all between the groups
        model = ols('Tm ~ C(Sample)', data=merged_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_val = anova_table['PR(>F)'].iloc[0]
        
        print(f"\n--- ANOVA Results ---")
        print(f"p-value: {p_val:.6f}")

        # 6. Post-Hoc Test (Tukey HSD)
        if p_val < 0.05:
            print("\nSignificant differences found (p < 0.05). Running Tukey HSD...")
            tukey = pairwise_tukeyhsd(endog=merged_df['Tm'], groups=merged_df['Sample'], alpha=0.05)
            print(tukey)
            
            # Save results to CSV
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.to_csv("dsf_stats_results.csv", index=False)
            print("\nFull statistical table saved to 'dsf_stats_results.csv'")
        else:
            print("\nNo statistically significant differences found between sample groups.")
       
       # 7. Export Results to CSV and PNG Image
        print("\nExporting results...")
        
        # Save CSV
        stats_summary.to_csv("dsf_summary_statistics.csv", index=False)
        
        # Create PNG Table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off') # Hide the graph axes
        
        # Format the numbers for the display table
        display_df = stats_summary.round(3)
        
        # Create the table
        tbl = ax.table(cellText=display_df.values, 
                       colLabels=display_df.columns, 
                       cellLoc='center', 
                       loc='center',
                       colColours=["#f2f2f2"] * len(display_df.columns))
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2) # Make rows taller for readability
        
        plt.title("DSF Summary Statistics", fontsize=14, pad=20)
        plt.savefig("dsf_stats_table.png", dpi=300, bbox_inches='tight')
        
        print("Success! Created 'dsf_summary_statistics.csv' and 'dsf_stats_table.png'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
