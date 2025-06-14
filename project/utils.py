import random
from sklearn.metrics import f1_score, precision_score, recall_score
import csv
import os
import numpy as np
import torch

def save_to_drive(content: str, filename: str, directory: str = "/content/drive/My Drive/MyProject") -> str:
    """
    Save text content to a file in the specified Google Drive directory.

    Args:
        content (str): The string content to be saved.
        filename (str): The desired name of the file (e.g., "metrics_log.txt").
        directory (str): Directory path on Google Drive. Will be created if it doesn't exist.

    Returns:
        str: Absolute path to the saved file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Data saved permanently to Google Drive at {file_path}")
    return file_path


def set_seed(seed: int):
    """
    Set the random seed across Python, NumPy, and PyTorch (CPU and GPU) for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def log_metrics_to_csv(csv_path: str, fold: int, iteration: int, loop, dataset, model, test_loader) -> int:
    """
    Evaluate the model on a test set and log key metrics to a CSV file.

    Args:
        csv_path (str): Path to the CSV file for metric logging. The file will be created if it doesn't exist.
        fold (int): Current fold index in cross-validation or experiment loop.
        iteration (int): Current iteration number of the active learning loop.
        loop (Any): Active learning loop object with expected attributes:
            - prev_labels (List[int], optional): Previously self-labeled indices.
            - current_oracle (int, optional): Number of new oracle-labeled samples in this iteration.
            - used_oracle_budget (int, optional): Total oracle query budget used so far.
        dataset (Any): DatasetManager object with labeled/unlabeled index attributes.
        model (Any): Model wrapper with:
            - model: the underlying PyTorch model.
            - device: device on which the model operates.
        test_loader (torch.utils.data.DataLoader): DataLoader for the held-out test set.

    Returns:
        int or str: The current used oracle budget after logging.
    """
    all_preds = []
    all_targets = []
    model.model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model.model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = 100 * (np.array(all_preds) == np.array(all_targets)).sum() / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')

    sl_samples = len(loop.prev_labels) if hasattr(loop, "prev_labels") else 'None'
    oracle_samples = loop.current_oracle if hasattr(loop, "current_oracle") else 'None'
    used_budget = loop.used_oracle_budget if hasattr(loop, "used_oracle_budget") else 'None'

    labeled_total = len(dataset.labeled_indices) + len(dataset.val_indices)
    unlabeled_total = len(dataset.unlabeled_indices)
    labeled_pct = 100 * labeled_total / (labeled_total + unlabeled_total)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Fold', 'Iteration',
                'Self-Labeled Samples', 'New Oracle Samples', 'Used Budget',
                'Total Labeled', 'Total Unlabeled', '% Labeled',
                'Accuracy', 'F1 Score', 'Precision', 'Recall'
            ])
        writer.writerow([
            fold, iteration,
            sl_samples, oracle_samples, used_budget,
            labeled_total, unlabeled_total, f"{labeled_pct:.2f}",
            f"{acc:.2f}", f"{f1:.4f}", f"{precision:.4f}", f"{recall:.4f}"
        ])

    return used_budget
