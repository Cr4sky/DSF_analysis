import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

# Suppress openpyxl style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def parse_args():
    parser = argparse.ArgumentParser(description="DSF Analysis: Multi-well, Replicates, and Filter")
    parser.add_argument("-f", "--file", required=True, help="Path to Data Excel file")
    parser.add_argument("-l", "--layout", help="Path to Layout Excel file (.xlsx)")
    parser.add_argument("-s", "--sheet", default="Melt Curve Raw Data", help="Data sheet name")
    parser.add_argument("--skip", type=int, default=45, help="Rows to skip in data file")
    parser.add_argument("--temp_col", default="Temperature", help="Temperature column name")
    parser.add_argument("--der_col", default="Derivative", help="Derivative column name")
    parser.add_argument("--well_col", default="Well Position", help="Well identifier column")
    parser.add_argument("--wells", default=None, help="Specific wells to plot (e.g., A1,A2)")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        # 1. Load the Main Data
        df = pd.read_excel(args.file, sheet_name=args.sheet, skiprows=args.skip, engine='openpyxl')
        df.columns = [str(c).strip() for c in df.columns]
        
        # 2. Hard Well Filter (Command line)
        if args.wells:
            target_wells = [w.strip() for w in args.wells.split(',')]
            df = df[df[args.well_col].isin(target_wells)]

        # 3. Load the Layout (as Excel)
        if args.layout:
            # Using read_excel here fixes the "Expected 2 fields" error
            layout_df = pd.read_excel(args.layout, engine='openpyxl')
            layout_df.columns = [str(c).strip() for c in layout_df.columns]
            
            # Match wells and merge
            df = df.merge(layout_df, on=args.well_col)
            group_col = 'Sample'
        else:
            group_col = args.well_col

        # 4. Plotting Setup
        plt.figure(figsize=(11, 7))
        samples = df[group_col].unique()
        # Colormap for better differentiation
        colors = plt.cm.get_cmap('Dark2', len(samples))
        line_styles = ['solid', 'dashed', 'dashdot', 'dotted']


        for i, sample in enumerate(samples):
            sample_df = df[df[group_col] == sample].copy()
            
            # Pivot: Rows=Temp, Cols=Wells, Values=Derivative
            pivot_df = sample_df.pivot(index=args.temp_col, columns=args.well_col, values=args.der_col)
            
            # Requirement: Plot Temperature vs Negative Derivative
            pivot_df = -pivot_df 
            
            l_style = line_styles[i % len(line_styles)]
            
            temps = pivot_df.index.values
            mean_curve = pivot_df.mean(axis=1)
            std_curve = pivot_df.std(axis=1)

            line_color = colors(i)
            plt.plot(temps, mean_curve, label=f'{sample}', linestyle=l_style, color=line_color, lw=2.5)
            
            # Shaded Standard Deviation for Replicates
            if pivot_df.shape[1] > 1:
                plt.fill_between(temps, mean_curve - std_curve, mean_curve + std_curve, 
                                 color=line_color, alpha=0.2)
            
            # Automatic Peak Detection and Labeling
            peak_idx = np.argmax(mean_curve.values)
            tm_temp = temps[peak_idx]
            plt.annotate(f'{tm_temp:.1f}°C', xy=(tm_temp, mean_curve.values[peak_idx]), 
                         xytext=(0, 12), textcoords='offset points', ha='center',
                         fontsize=10, color=line_color, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color=line_color, lw=1))

        plt.title(f'DSF Analysis: {args.sheet}', fontsize=14, pad=20)
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Negative Derivative ($-dF/dT$)', fontsize=12)
        plt.legend(title="Sample Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        plt.savefig(f"dsf_grouped_analysis.png", dpi=300)
        print(f"Success! Analyzed {len(samples)} groups.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
