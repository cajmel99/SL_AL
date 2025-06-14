from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
from model import SemiSupervisedModel
from data_manager import DatasetManager
from strategies import SelfLabelingLoop
from utils import log_metrics_to_csv

if __name__ == "__main__":
    n_folds = 5
    n_iteration = 10
    all_initial_labels = 500
    n_initial_labels = int(0.8*all_initial_labels)
    val_split_size = int(0.2*all_initial_labels)
    batch_size = 64
    confidence_threshold = 0.95
    max_oracle_budget = 5000
    n_epochs = 10
    csv_path = f"SL_{all_initial_labels}_{max_oracle_budget}.csv"
    oracle_correction_size = 400 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    targets = np.array(full_dataset.targets)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n================== FOLD {fold_idx + 1} ==================")

        train_targets = targets[train_idx]

        val_indices, remaining_idx = train_test_split(
            train_idx,
            train_size=val_split_size,
            stratify=train_targets,
            random_state=fold_idx
        )

        # DL and DU from train_set
        remaining_targets = targets[remaining_idx]
        labeled_indices, unlabeled_indices = train_test_split(
            remaining_idx,
            train_size=n_initial_labels,
            stratify=remaining_targets,
            random_state=fold_idx
        )

        dataset = DatasetManager(
            batch_size=batch_size,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            val_indices=val_indices,
            test_indices=test_idx
        )

        # Update Budget
        initial_oracle_cost = len(labeled_indices) + len(val_indices)
        current_max_oracle_budget = max_oracle_budget # Keep the total budget

        model = SemiSupervisedModel()
        loop = SelfLabelingLoop(model, dataset, confidence_threshold, current_max_oracle_budget, used_oracle_budget=initial_oracle_cost, oracle_correction_size=oracle_correction_size)

        train_loader, val_loader, unlabeled_loader, test_loader = dataset.get_loaders()

        model.train(train_loader, val_dataloader=val_loader, epochs=n_epochs)

        log_metrics_to_csv(csv_path=csv_path, fold=fold_idx + 1,
                                    iteration=0,
                                    loop=loop,
                                    dataset=dataset,
                                    model=model,
                                    test_loader=val_loader,
                                )

        # Iteration loop
        for i in range(n_iteration):
            if loop.used_oracle_budget >= loop.max_oracle_budget:
                print(f"Oracle budget exceeded before iteration {i+1}. Stopping early.")
                break

            print(f"\nIteration {i+1}/{n_iteration} (Fold {fold_idx + 1})")
            loop.run_iteration(val_loader=val_loader)  

            # Get updated loaders 
            train_loader, _, _, _ = dataset.get_loaders()
            model.train(train_loader, val_loader, epochs=n_epochs)

            print(f"\nFinal stats for Fold {fold_idx + 1}, Iteration {i+1}")
            print(f"Labeled samples: {len(dataset.labeled_indices)}")
            print(f"Unlabeled samples: {len(dataset.unlabeled_indices)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print(f"Oracle-labeled samples (total used): {loop.used_oracle_budget}")
            print(f"Raming budget: {loop.max_oracle_budget - loop.used_oracle_budget}")

            # Save results for the current iteration
            log_metrics_to_csv(csv_path=csv_path, fold=fold_idx + 1,
                                    iteration=i + 1,
                                    loop=loop,
                                    dataset=dataset,
                                    model=model,
                                    test_loader=val_loader,
                                )

        # Evaluate on the test set at the end of all iterations for this fold
        final_acc = model.evaluate(test_loader)
        print(f"\nFinal Test Accuracy for Fold {fold_idx+1}: {final_acc:.2f}%")
        fold_results.append(final_acc)

    print("\nFINAL CV RESULTS")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Mean: {np.mean(fold_results):.2f}%, Std: {np.std(fold_results):.2f}%")

    # Save the path to the CSV file in Google Drive
    drive_file_path = "/content/drive/My Drive/MyProject/sl_al_correction_results.txt" # Changed filename
    drive_directory = "/content/drive/My Drive/MyProject"

    if not os.path.exists(drive_directory):
        os.makedirs(drive_directory)

    with open(drive_file_path, "w") as f:
        f.write(csv_path)

    print(f"Results saved at {drive_file_path}")

