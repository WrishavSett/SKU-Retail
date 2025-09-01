import pandas as pd
import os
import glob
import argparse

def compute_metrics(transaction_csv):
    """
    Compute Top-1, Top-2, Top-3 accuracy + classification metrics
    (precision, recall, specificity, type I/II error)
    based on whether the true code is in the Top-3 predictions.
    """
    df = pd.read_csv(transaction_csv)
    df_subset = df[['t_NEW_CODES', 'rank', 'matched_itemcode']]
    grouped = df_subset.groupby('t_NEW_CODES')

    total_transactions = len(grouped)
    top1_correct = top2_correct = top3_correct = 0

    # Confusion matrix components (Top-3 presence)
    TP = FP = FN = TN = 0

    for t_new_code, group in grouped:
        group_sorted = group.sort_values('rank')
        matched_codes = group_sorted['matched_itemcode'].tolist()

        # --- Top-k accuracy ---
        if t_new_code == matched_codes[0]:
            top1_correct += 1
        if t_new_code in matched_codes[:2]:
            top2_correct += 1
        if t_new_code in matched_codes[:3]:
            top3_correct += 1

        # --- Confusion Matrix for Top-3 ---
        if t_new_code in matched_codes[:3]:  # Correctly predicted within Top-3
            TP += 1
            FP += (3 - 1)  # the other 2 predictions are wrong
        else:
            FN += 1
            FP += 3       # all 3 predictions wrong

    # TN remains ambiguous in multi-class, leave at 0
    TN = 0

    # --- Compute Metrics ---
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    type1_error = FP / (FP + TN) if (FP + TN) > 0 else 0
    type2_error = FN / (FN + TP) if (FN + TP) > 0 else 0

    return {
        'File': os.path.basename(transaction_csv),
        'Total Transactions': total_transactions,
        'Top-1 Accuracy': top1_correct / total_transactions,
        'Top-2 Accuracy': top2_correct / total_transactions,
        'Top-3 Accuracy': top3_correct / total_transactions,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'Type I Error': type1_error,
        'Type II Error': type2_error
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

        # --- Print metrics ---
        print(f"Total transactions : {metrics['Total Transactions']}")
        print(f"Top-1 Accuracy     : {metrics['Top-1 Accuracy']:.4f}")
        print(f"Top-2 Accuracy     : {metrics['Top-2 Accuracy']:.4f}")
        print(f"Top-3 Accuracy     : {metrics['Top-3 Accuracy']:.4f}")
        print(f"Precision (Top-3)  : {metrics['Precision']:.4f}")
        print(f"Recall (Top-3)     : {metrics['Recall']:.4f}")
        print(f"Specificity (Top-3): {metrics['Specificity']:.4f}")
        print(f"Type I Error       : {metrics['Type I Error']:.4f}")
        print(f"Type II Error      : {metrics['Type II Error']:.4f}")

        all_metrics.append(metrics)

    # Save all metrics to a single CSV
    output_csv = './temp/metrics_all_transactions.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(output_csv, index=False)
    print(f"\nAll metrics saved to: {output_csv}")

if __name__ == "__main__":
    main()
