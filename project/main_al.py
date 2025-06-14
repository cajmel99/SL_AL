from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
from model import SemiSupervisedModel
from data_manager import DatasetManager
from strategies import ActiveLearningLoop
from utils import log_metrics_to_csv

if __name__ == "__main__":
    n_folds = 5
    n_iteration = 10
    all_initial_labels = 2000
    initial_labels = int(0.8*all_initial_labels)
    val_split_size = int(0.2*all_initial_labels)

    batch_size = 64
    acquisition_size = 800  
    csv_path = f"AL_{all_initial_labels}_{all_initial_labels+n_iteration*acquisition_size}.csv"
    epochs = 10 # Number of training epochs per iteration
    # early_stopping_patience = 7 # Patience for early stopping

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

        remaining_targets = targets[remaining_idx]
        labeled_indices, unlabeled_indices = train_test_split(
            remaining_idx,
            train_size=initial_labels,
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
        loop = ActiveLearningLoop(model, dataset, acquisition_size, strategy="random")

        train_loader, val_loader, _, test_loader = dataset.get_loaders()
        # print(f"Fold {fold_idx+1}, Labeled={len(labeled_indices)}, "
        #       f"Unlabeled={len(unlabeled_indices)}, Val={len(val_loader.dataset)}")

        model.train(train_loader, val_loader, epochs=epochs) 
        log_metrics_to_csv(
                csv_path=csv_path,
                fold=fold_idx + 1,
                iteration=0,
                loop=loop,
                dataset=dataset,
                model=model,
                test_loader=val_loader
            )
        for i in range(n_iteration):
            if len(dataset.unlabeled_indices) < acquisition_size:
                print(f"Not enough unlabeled samples to continue. Stopping.")
                break

            print(f"\nIteration {i + 1}/{n_iteration} (Fold {fold_idx + 1})")
            loop.run_iteration()

            train_loader, val_loader, _, _ = dataset.get_loaders() # Get updated val_loader
            model.train(train_loader, val_loader, epochs=epochs) 

            print(f"Stats after iteration {i+1}:")
            print(f"Labeled: {len(dataset.labeled_indices)} samples")
            print(f"Unlabeled: {len(dataset.unlabeled_indices)} samples")

            # Log metrics on validation set
            log_metrics_to_csv(
                csv_path=csv_path,
                fold=fold_idx + 1,
                iteration=i + 1,
                loop=loop,
                dataset=dataset,
                model=model,
                test_loader=val_loader
            )


        final_acc = model.evaluate(test_loader)
        fold_results.append(final_acc)

    print("\nFINAL CROSS-VALIDATION RESULTS")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Mean Accuracy: {np.mean(fold_results):.2f}%, Std: {np.std(fold_results):.2f}%")