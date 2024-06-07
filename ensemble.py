import sys
import pandas as pd
import os
from scipy.stats import mode
import numpy as np
import zipfile
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))

MODAL_FILEPATH = os.path.join(current_directory, 'modal_results')
ENSEMBLES_OUTPUT = os.path.join(current_directory, 'ensembles')

def create_zip_from_csvs(csv_files, output, dataset, models):
    if not csv_files:
        print("No CSV files provided.")
        return
    
    zip_file_name = os.path.join(output, f"{dataset}-submission-{'-'.join(models)}.zip")

    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)
    
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        for csv_file in csv_files:
            if os.path.isfile(csv_file):
                arcname = os.path.join("predictions", os.path.basename(csv_file))
                zipf.write(csv_file, arcname)
            else:
                print(f"File not found: {csv_file}")

    print(f"Submission file saved to {zip_file_name}")
    

def read_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            file.readline()
            df = pd.read_csv(file, header=None, names=['id', 'prediction'])
            return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def validate_ids(dfs):
    ids_sets = [set(df['id']) for df in dfs]
    common_ids = set.intersection(*ids_sets)
    if not all(ids == common_ids for ids in ids_sets):
        print("Error: The CSV files do not contain the same IDs.")
        sys.exit(1)
    return list(common_ids)

def run(csv_files, output, task):
    # Read all CSV files
    dfs = [read_csv(file) for file in csv_files]

    # Validate IDs
    common_ids = validate_ids(dfs)
    common_ids.sort()

    # Sort dataframes by id
    for df in dfs:
        df.set_index('id', inplace=True)
        df.sort_index(inplace=True)
        # print(df)

    # Combine predictions
    combined_df = pd.DataFrame(index=common_ids)
    if task == "regression":
        combined_df['ensemble_prediction'] = sum(df['prediction'] for df in dfs) / len(dfs)
    else:
        # MEDIAN version
        combined_df['ensemble_prediction'] = np.round(np.median([df['prediction'] for df in dfs], axis=0)).astype(int)

        # MODE version
        # combined_df['ensemble_prediction'] = mode([df['prediction'] for df in dfs], axis=0, keepdims=False).mode

    if task == "regression":
        combined_df['ensemble_prediction'] = combined_df['ensemble_prediction'].clip(0, 5)

    # print(combined_df)
    
    # Reset index to make 'id' a column again
    combined_df.reset_index(inplace=True)
    
    # Save the combined predictions
    if os.path.exists(output):
        os.remove(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    combined_df.to_csv(output, index=False)
    print("Ensemble predictions saved to ", output)


def main(models, dataset):
    output_dir = os.path.join(ENSEMBLES_OUTPUT, "-".join(models['all']))
    outputs = []

    for task in ["sst", "para", "sts"]:
        csv_files = [os.path.join(MODAL_FILEPATH, model, "predictions", f"{task}-{dataset}-output.csv") for model in models[task]]
        output = os.path.join(output_dir, f"{task}-{dataset}-output.csv")
        outputs.append(output)
        run(csv_files, output, "regression" if task == "sts" else "classification")

    create_zip_from_csvs(outputs, output_dir, dataset, models['all'])

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple models")
    parser.add_argument("models", nargs="+", help="List of models to ensemble")
    parser.add_argument("--sst_models", nargs="+", help="List of SST models to ensemble", default=[])
    parser.add_argument("--para_models", nargs="+", help="List of Para models to ensemble", default=[])
    parser.add_argument("--sts_models", nargs="+", help="List of STS models to ensemble", default=[])
    parser.add_argument("--dataset", default="dev", help="Dataset to ensemble (dev or test)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    models = {
        "all": args.models + args.sst_models + args.para_models + args.sts_models,
        "sst": args.sst_models + args.models,
        "para": args.para_models + args.models,
        "sts": args.sts_models + args.models
    }

    main(models, args.dataset)

