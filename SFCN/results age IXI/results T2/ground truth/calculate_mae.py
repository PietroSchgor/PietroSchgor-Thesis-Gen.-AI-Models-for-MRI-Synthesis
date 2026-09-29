import pandas as pd
import os

def main():
    file_path = r"F:\tesi\SFCN\results age IXI\results T2\ground truth\ixi_cross_validation_predictions.txt"
    output_path = r"F:\tesi\SFCN\results age IXI\results T2\ground truth\mae_results.txt"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # MAE per fold (assuming 'Site' represents the fold)
    mae_per_site = df.groupby('Site')['Absolute_Error'].mean()

    # Total MAE
    total_mae = df['Absolute_Error'].mean()

    with open(output_path, 'w') as f:
        f.write("MAE per fold (Site):\n")
        for site, mae in mae_per_site.items():
            f.write(f"{site}: {mae:.4f}\n")
        f.write("\n")
        f.write(f"Total MAE: {total_mae:.4f}\n")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
