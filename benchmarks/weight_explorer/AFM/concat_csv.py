# concatenate all specified csv files in a directory
import os
import pandas as pd

import glob

def concat_csv(csv_dir, output_csv):
    csv_files = glob.glob(os.path.join(csv_dir, "weight_norm_diff_experts_d*.csv"))
    print("Concatenating csv files:", csv_files)
    input("Press Enter to continue, or Ctrl+C to cancel.")
    df = pd.concat([pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files])
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=str, help="Directory containing csv files to concatenate")
    parser.add_argument("output_csv", type=str, help="Path to save concatenated csv file")
    args = parser.parse_args()
    concat_csv(args.csv_dir, args.output_csv)