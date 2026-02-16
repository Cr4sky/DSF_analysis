import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import warnings

# Suppress openpyxl style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def parse_args():
    parser = argparse.ArgumentParser(description="DSF Statistical Kd Analysis")
    parser.add_argument("-f", "--file", required=True, help="Path to Data Excel file")
    parser.add_argument("-l", "--layout", required=True, help="Path to Layout Excel file (.xlsx)")
    parser.add_argument("-s", "--sheet", default="Melt Curve Raw Data", help="Data sheet name")
    parser.add_argument("--skip", type=int, default=45, help="Rows to skip in data file")
    parser.add_argument("--well_col", default="Well Position", help="Well identifier column")
    return parser.parse_args()

def hill_equation(L, Kd, Top, Bottom):
    """1:1 Binding Model: Tm = Bottom + (Top - Bottom) * L / (Kd + L)"""
    return Bottom + (Top - Bottom) * L / (Kd + L)

def calculate_statistics(x, y, popt, pcov):
    """
    Calculates R-squared, Standard Errors, 95% Confidence Intervals, 
    and the F-test (Model vs. Null Hypothesis).
    """
    N = len(y)      # Number of data points
    P = len(popt)   # Number of parameters (Kd, Top, Bottom)
    
    df_resid = N - P  # Degrees of freedom for residuals
    
    # 1. Residual Sum of Squares (RSS)
    residuals = y - hill_equation(x, *popt)
    ss_res = np.sum(residuals**2)
    
    # 2. Total Sum of Squares (TSS) - Null Hypothesis (flat line at mean)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    # 3. R-squared
    r_squared = 1 - (ss_res / ss_tot)
    
    # 4. F-test (Comparing Hill Fit to a flat line)
    # df1 is difference in params (3-1=2); df2 is residual df
    if ss_res > 0 and df_resid > 0:
        f_stat = ((ss_tot - ss_res) / (P - 1)) / (ss_res / df_resid)
        p_val_f = stats.f.sf(f_stat, (P - 1), df_resid)
    else:
        f_stat, p_val_f = 0.0, 1.0

    # 5. Standard Errors (perr) and 95% Confidence Intervals (CI)
    perr = np.sqrt(np.diag(pcov))
    t_val = stats.t.ppf(0.975, df_resid) if df_resid > 0 else 0
    
    ci_low = popt - (t_val * perr)
    ci_high = popt + (t_val * perr)
    
    return r_squared, perr, ci_low, ci_high, f_stat, p_val_f

def main():
    args = parse_args()

    try:
        # 1. Load Data and Layout
        df = pd.read_excel(args.file, sheet_name=args.sheet, skiprows=args.skip, engine='openpyxl')
        layout = pd.read_excel(args.layout, engine='openpyxl')
        df = df.merge(layout, on=args.well_col)

        stats_list = []
        plt.figure(figsize=(10, 6))
        samples = df['Sample'].unique()
        colors = plt.cm.get_cmap('Dark2', len(samples))

        # 2. Analyze per Sample Group
        for i, sample_name in enumerate(samples):
            sample_group = df[df['Sample'] == sample_name]
            well_results = []

            for well in sample_group[args.well_col].unique():
                well_df = sample_group[sample_group[args.well_col] == well]
                # Peak detection using negative derivative
                neg_der = -well_df['Derivative']
                tm = well_df.iloc[np.argmax(neg_der.values)]['Temperature']
                conc = float(well_df['Concentration'].iloc[0])
                well_results.append({'Conc': conc, 'Tm': tm})
            
            res_df = pd.DataFrame(well_results).sort_values('Conc')
            
            # --- IMPORTANT: xdata and ydata are defined here ---
            xdata, ydata = res_df['Conc'], res_df['Tm']

            # 3. Non-Linear Regression with Bounds
            p0 = [xdata.median() or 1.0, ydata.max(), ydata.min()]
            bounds = ([0, ydata.min(), ydata.min() - 5], [5000, 110, ydata.max() + 5])
            
            popt, pcov = curve_fit(hill_equation, xdata, ydata, p0=p0, bounds=bounds)
            
            # --- CALL STATISTICS HERE (Inside the loop) ---
            r2, perr, ci_low, ci_high, f_stat, p_val_f = calculate_statistics(xdata, ydata, popt, pcov)
            
            stats_list.append({
                'Sample': sample_name, 
                'Kd (uM)': popt[0], 
                'Kd_SE': perr[0],
                'Kd_95CI_Low': ci_low[0], 
                'Kd_95CI_High': ci_high[0], 
                'R_squared': r2,
                'F_statistic': f_stat,
                'p_value_F': p_val_f
            })

            # 4. Plotting
            color = colors(i)
            plt.scatter(xdata, ydata, color=color, alpha=0.6)
            x_fit = np.logspace(np.log10(xdata.min() or 0.1), np.log10(xdata.max()), 100)
            plt.plot(x_fit, hill_equation(x_fit, *popt), color=color, lw=2, 
                     label=f'{sample_name} ($R^2$: {r2:.3f}, p: {p_val_f:.4f})')

        # 5. Output Results
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv("dsf_stats_comprehensive_results.csv", index=False)
        
        plt.xscale('log')
        plt.xlabel(r'Concentration ($\mu$M)')
        plt.ylabel(r'$T_m$ (°C)')
        plt.title('Dose-Response Fit with F-Test and Confidence Intervals')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("dsf_statistical_plot.png", dpi=300)
        
        print("\n--- Fit Statistics Summary ---")
        print(stats_df[['Sample', 'Kd (uM)', 'Kd_95CI_Low', 'Kd_95CI_High', 'p_value_F']])

    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    main()
