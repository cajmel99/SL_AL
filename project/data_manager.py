from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms


class PseudoLabeledDataset(Dataset):
    """
    A wrapper around a PyTorch Dataset that injects pseudo-labels where available.

    Attributes:
        original_dataset (torch.utils.data.Dataset): The original labeled dataset.
        pseudo_labels_dict (dict): A dictionary mapping sample indices to pseudo-labels.
    """

    def __init__(self, original_dataset, pseudo_labels_dict):
        """
        Initialize the pseudo-labeled dataset.

        Args:
            original_dataset (torch.utils.data.Dataset): The base dataset containing input images and labels.
            pseudo_labels_dict (dict): Dictionary with pseudo-labels, where keys are dataset indices
                                       and values are the corresponding pseudo-labels.
        """
        self.original_dataset = original_dataset
        self.pseudo_labels_dict = pseudo_labels_dict

    def __getitem__(self, index):
        """
        Retrieve the image and corresponding label for a given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where label is either a pseudo-label (if available)
                   or the original label.
        """
        img, _ = self.original_dataset[index]
        label = self.pseudo_labels_dict.get(index, self.original_dataset.targets[index])
        return img, label

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.original_dataset)


class DatasetManager:
    """
    Handles dataset partitioning and DataLoader creation for semi-supervised learning workflows.

    Supports splitting the CIFAR-10 dataset into labeled, unlabeled, validation, and test sets.
    Applies standard normalization and pseudo-label integration.

    Attributes:
        batch_size (int): Batch size for the DataLoaders.
        labeled_indices (np.ndarray): Indices of labeled data samples.
        unlabeled_indices (np.ndarray): Indices of unlabeled data samples.
        val_indices (np.ndarray or None): Indices for the validation set.
        test_indices (np.ndarray): Indices for the test set.
        pseudo_labels_dict (dict): Pseudo-labels associated with specific sample indices.
        trainset (torchvision.datasets.CIFAR10): The base training dataset.
        transform (torchvision.transforms.Compose): Transformations applied to the data.
    """

    def __init__(self,
                 batch_size,
                 labeled_indices,
                 unlabeled_indices,
                 val_indices,
                 test_indices,
                 pseudo_labels_dict=None):
        
        self.batch_size = batch_size
        self.labeled_indices = np.array(labeled_indices, dtype=np.int64)
        self.unlabeled_indices = np.array(unlabeled_indices, dtype=np.int64)
        self.val_indices = np.array(val_indices, dtype=np.int64) if val_indices is not None else None
        self.test_indices = np.array(test_indices, dtype=np.int64)
        self.pseudo_labels_dict = pseudo_labels_dict if pseudo_labels_dict is not None else {}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)

    def get_loaders(self):
        """
        Generate DataLoaders for labeled training, validation, unlabeled, and test data.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
                - train_loader: DataLoader for labeled training data (with pseudo-labels if available).
                - val_loader: DataLoader for validation data.
                - unlabeled_loader: DataLoader for unlabeled data (for use in semi-supervised learning).
                - test_loader: DataLoader for the test set.
        """
        labeled_dataset = PseudoLabeledDataset(self.trainset, self.pseudo_labels_dict)

        train_loader = DataLoader(
            Subset(labeled_dataset, self.labeled_indices),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            Subset(self.trainset, self.val_indices),
            batch_size=self.batch_size,
            shuffle=False
        ) if self.val_indices is not None else None

        unlabeled_loader = DataLoader(
            Subset(self.trainset, self.unlabeled_indices),
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            Subset(self.trainset, self.test_indices),
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, unlabeled_loader, test_loader

    def update_indices(self, new_indices):
        """
        Move new indices from the unlabeled pool to the labeled pool.

        Args:
            new_indices (list[int]): Sample indices to mark as newly labeled.
        """
        new_indices = np.array(new_indices, dtype=np.int64)
        self.labeled_indices = np.concatenate([self.labeled_indices, new_indices])
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, new_indices)
