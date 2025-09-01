import pandas as pd
import os
import glob
import argparse

def compute_metrics(transaction_csv):
    """
    Compute Top-1, Top-2, Top-3 accuracy for a given transaction match CSV.
    Returns a dictionary with metrics and file name.
    """
    df = pd.read_csv(transaction_csv)
    df_subset = df[['t_NEW_CODES', 'rank', 'matched_itemcode']]
    grouped = df_subset.groupby('t_NEW_CODES')

    total_transactions = len(grouped)
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0

    for t_new_code, group in grouped:
        group_sorted = group.sort_values('rank')
        matched_codes = group_sorted['matched_itemcode'].tolist()

        if t_new_code == matched_codes[0]:
            top1_correct += 1
        if t_new_code in matched_codes[:2]:
            top2_correct += 1
        if t_new_code in matched_codes[:3]:
            top3_correct += 1

    return {
        'File': os.path.basename(transaction_csv),
        'Total Transactions': total_transactions,
        'Top-1 Accuracy': top1_correct / total_transactions,
        'Top-2 Accuracy': top2_correct / total_transactions,
        'Top-3 Accuracy': top3_correct / total_transactions
    }

def main():
    parser = argparse.ArgumentParser(description="Compute transaction matching metrics")
    parser.add_argument("--file", type=str, help="Path to a single transaction match CSV")
    parser.add_argument("--folder", type=str, help="Path to a folder containing transaction match CSVs")
    args = parser.parse_args()

    transaction_files = []
    if args.file:
        if os.path.exists(args.file):
            transaction_files = [args.file]
        else:
            print(f"Error: File not found -> {args.file}")
            return
    elif args.folder:
        folder_path = args.folder
        # Only include files that contain 'transaction_matches' in their filename
        transaction_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if "transaction_matches" in os.path.basename(f)]
        if not transaction_files:
            print(f"No transaction match CSVs found in folder: {folder_path}")
            return
    else:
        print("Error: Please provide either --file or --folder")
        return

    all_metrics = []
    for file_path in transaction_files:
        print(f"\nProcessing metrics for: {file_path}")
        metrics = compute_metrics(file_path)
        print(f"Total transactions: {metrics['Total Transactions']}")
        print(f"Top-1 Accuracy  : {metrics['Top-1 Accuracy']:.4f}")
        print(f"Top-2 Accuracy  : {metrics['Top-2 Accuracy']:.4f}")
        print(f"Top-3 Accuracy  : {metrics['Top-3 Accuracy']:.4f}")
        all_metrics.append(metrics)

    # Save all metrics to a single CSV
    output_csv = './temp/metrics_all_transactions.csv'
    pd.DataFrame(all_metrics).to_csv(output_csv, index=False)
    print(f"\nAll metrics saved to: {output_csv}")

if __name__ == "__main__":
    main()
