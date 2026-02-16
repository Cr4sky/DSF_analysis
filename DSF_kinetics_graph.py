import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # Corrected function name
import warnings

# Suppress openpyxl style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def hill_equation(L, Kd, Top, Bottom):
    """Simple 1:1 binding model for Tm shift."""
    return Bottom + (Top - Bottom) * L / (Kd + L)

def parse_args():
    parser = argparse.ArgumentParser(description="DSF Kd Analysis: Tm vs Concentration")
    parser.add_argument("-f", "--file", required=True, help="Data Excel file")
    parser.add_argument("-l", "--layout", required=True, help="Layout file with 'Well Position', 'Sample', 'Concentration'")
    parser.add_argument("-s", "--sheet", default="Melt Curve Raw Data", help="Data sheet name")
    parser.add_argument("--skip", type=int, default=45, help="Rows to skip in data file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # 1. Load Data
        df = pd.read_excel(args.file, sheet_name=args.sheet, skiprows=args.skip)
        layout = pd.read_excel(args.layout)
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        layout.columns = [str(c).strip() for c in layout.columns]
        
        # Merge data with layout
        df = df.merge(layout, on="Well Position")

        plt.figure(figsize=(10, 6))
        samples = df['Sample'].unique()
        colors = plt.cm.get_cmap('tab10', len(samples))

        # 2. Process each Sample separately
        for i, sample_name in enumerate(samples):
            sample_df = df[df['Sample'] == sample_name]
            well_data = []

            # Extract Tm for each concentration point (well)
            for well in sample_df['Well Position'].unique():
                well_df = sample_df[sample_df['Well Position'] == well]
                neg_der = -well_df['Derivative']
                
                # Identify Tm at the peak of the negative derivative
                peak_idx = np.argmax(neg_der.values)
                tm = well_df.iloc[peak_idx]['Temperature']
                conc = float(well_df['Concentration'].iloc[0])
                well_data.append({'Conc': conc, 'Tm': tm})

            res_df = pd.DataFrame(well_data).sort_values('Conc')
            
            # 3. Fit the Curve (Non-linear Regression)
            # Initial guesses: [Kd, Top, Bottom]
            p0 = [res_df['Conc'].median() or 1.0, res_df['Tm'].max(), res_df['Tm'].min()]
            
            try:
                popt, _ = curve_fit(hill_equation, res_df['Conc'], res_df['Tm'], p0=p0)
                kd_val = popt[0]
                
                # Generate fit line
                min_c = res_df[res_df['Conc'] > 0]['Conc'].min() * 0.5
                x_fit = np.logspace(np.log10(min_c), np.log10(res_df['Conc'].max()), 100)
                y_fit = hill_equation(x_fit, *popt)

                # 4. Plotting using raw f-strings to avoid SyntaxErrors
                color = colors(i)
                plt.scatter(res_df['Conc'], res_df['Tm'], color=color, alpha=0.7)
                plt.plot(x_fit, y_fit, color=color, lw=2, 
                         label=fr'{sample_name} ($K_d$: {kd_val:.2f} $\mu$M)')
            except Exception as fit_error:
                print(f"Could not fit curve for {sample_name}: {fit_error}")

        # Final Formatting
        plt.xscale('log')
        plt.xlabel(r'Concentration ($\mu$M)', fontsize=12)
        plt.ylabel(r'Melting Temperature ($T_m$ °C)', fontsize=12)
        plt.title('DSF Dose-Response: Kd Determination', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        plt.savefig("dsf_kd_results.png", dpi=300)
        print("Analysis complete. Plot saved as dsf_kd_results.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
