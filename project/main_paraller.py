import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets, transforms
from model import SemiSupervisedModel
from data_manager import DatasetManager
from strategies import HybridActiveSelfLabelingLoop
from utils import log_metrics_to_csv

if __name__ == "__main__":
    n_folds = 5
    n_iteration = 20
    all_initial_labels = 500
    n_initial_labels = int(0.8*all_initial_labels)
    val_split_size = int(0.2*all_initial_labels)
    batch_size = 64
    confidence_threshold = 0.95
    max_oracle_budget = 2500 # Total budget
    n_epochs = 10

    # Parameters for the hybrid loop
    k_oracle_per_iter = 200 # Number of samples to query from oracle per iteration (AL part)
    k_selflabel_per_iter = 200 # Number of samples to self-label per iteration (SL part)
    oracle_correction_size = 200 # Samples to correct with oracle if accuracy drops
    use_oracle_correction = True 

    csv_path = f"SL_AL_correction_hybrid_{all_initial_labels}_{max_oracle_budget}.csv"

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

        # DL and DU from 
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

        model = SemiSupervisedModel()

        # Initialize the HybridActiveSelfLabelingLoop
        initial_oracle_cost = len(labeled_indices) + len(val_indices)
        loop = HybridActiveSelfLabelingLoop(
            model_wrapper=model,
            dataset_manager=dataset,
            k_oracle_per_iter=k_oracle_per_iter,
            k_selflabel_per_iter=k_selflabel_per_iter,
            confidence_threshold=confidence_threshold,
            max_oracle_budget=max_oracle_budget,
            oracle_correction_size=oracle_correction_size,
            use_oracle_correction=use_oracle_correction
        )

        # Set the initial used budget in the loop
        loop.used_oracle_budget = initial_oracle_cost
        loop.oracle_budget = initial_oracle_cost 

        train_loader, val_loader, unlabeled_loader, test_loader = dataset.get_loaders()

        print(f"Labeled: {len(dataset.labeled_indices)} samples")
        print(f"Unlabeled: {len(dataset.unlabeled_indices)} samples")

        model.train(train_loader, val_loader, epochs=n_epochs)

        loop.prev_accuracy = model.evaluate(val_loader)
        print(f"Initial Validation Accuracy: {loop.prev_accuracy:.2f}%")

        log_metrics_to_csv(csv_path=csv_path, fold=fold_idx + 1,
                                    iteration=0,
                                    loop=loop, 
                                    dataset=dataset,
                                    model=model,
                                    test_loader=val_loader, 
                                )

        # Iteration loop
        for i in range(n_iteration):
            print(f"\n================== Iteration {i+1}/{n_iteration} (Fold {fold_idx + 1}) ==================")

            if loop.used_oracle_budget >= loop.max_oracle_budget:
                print(f"Oracle budget exceeded before iteration {i+1}. Stopping early.")
                break
            if len(dataset.unlabeled_indices) == 0:
                 print(f"No unlabeled samples left at iteration {i+1}. Stopping early.")
                 break

            # Run hybrid
            loop.run_iteration(val_loader=val_loader)

            # Get updated loaders
            train_loader, val_loader, _, _ = dataset.get_loaders()

            model.train(train_loader, val_loader, epochs=n_epochs)

            print(f"\nStats after iteration {i+1}:")
            print(f"Labeled samples: {len(dataset.labeled_indices)}")
            print(f"Unlabeled samples: {len(dataset.unlabeled_indices)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print(f"Oracle-labeled samples (total used): {loop.used_oracle_budget}")
            print(f"Remaining budget: {loop.max_oracle_budget - loop.used_oracle_budget}")
            print(f"Samples self-labeled in this iter: {loop.current_selflabeled_in_iter}")
            print(f"Samples oracle-labeled in this iter: {loop.current_oracle_in_iter}")

            log_metrics_to_csv(csv_path=csv_path, fold=fold_idx + 1,
                                    iteration=i + 1,
                                    loop=loop, 
                                    dataset=dataset,
                                    model=model,
                                    test_loader=val_loader, 
                                )

        # Evaluate on the test_set
        print(f"\nEvaluating on test set for Fold {fold_idx+1}...")
        final_acc = model.evaluate(test_loader)
        print(f"\nFinal Test Accuracy for Fold {fold_idx+1}: {final_acc:.2f}%")
        fold_results.append(final_acc)

    print("\nFINAL CROSS-VALIDATION RESULTS")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Mean Accuracy: {np.mean(fold_results):.2f}%, Std: {np.std(fold_results):.2f}%")

    # Confirm where the results were saved
    print(f"\nResults saved permanently to Google Drive (or local path) at: {csv_path}")