import numpy as np
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

class SelfLabelingLoop:
    def __init__(self, model_wrapper, dataset_manager, confidence_threshold, max_oracle_budget, used_oracle_budget, use_oracle=False, oracle_correction_size=800):
        self.model_wrapper = model_wrapper
        self.dataset_manager = dataset_manager
        self.confidence_threshold = confidence_threshold
        self.max_oracle_budget = max_oracle_budget
        self.used_oracle_budget = used_oracle_budget
        self.prev_labels = {}
        self.prev_accuracy = 0
        self.use_oracle = use_oracle
        self.current_oracle = 0
        self.oracle_correction_size = oracle_correction_size
        self.previous_confidences = {}
        self.corrected_count = 0
        self.n_iteration_from_quering = 1
        self.n_to_query = 0
        self.limit_sl = 100000

    def run_iteration(self, val_loader):
        _, _, unlabeled_loader, _ = self.dataset_manager.get_loaders()

        confidences, predictions = self.model_wrapper.predict_confidences(unlabeled_loader)

        selected_samples = [i for i, conf in enumerate(confidences) if np.max(conf) >= self.confidence_threshold]

        # Limit the sl
        selected_samples = selected_samples[:self.limit_sl]

        current_labels = {}  

        for i in selected_samples:
            dataset_index = self.dataset_manager.unlabeled_indices[i]
            input_tensor, true_label = self.dataset_manager.trainset[dataset_index]
            predicted_label = predictions[i]
            confidence = np.max(confidences[i])

            current_labels[dataset_index] = {
                'index': dataset_index,
                'pseudo_label': predicted_label,
                'oracle_label': true_label,
                'confidence': confidence
            }

            # Apply pseudo-label 
            self.dataset_manager.pseudo_labels_dict[dataset_index] = predicted_label
            # Update DL
            self.dataset_manager.labeled_indices = np.append(self.dataset_manager.labeled_indices, dataset_index)


        # Remove current samples from unlabeled
        self.dataset_manager.unlabeled_indices = np.setdiff1d(
            self.dataset_manager.unlabeled_indices,
            np.array(list(current_labels.keys()), dtype=int)
        )

        # Evaluate
        test_accuracy = self.model_wrapper.evaluate(val_loader)
        if test_accuracy <= self.prev_accuracy:
            self.n_to_query = self.oracle_correction_size * self.n_iteration_from_quering
            if self.use_oracle:
                print("Accuracy dropped! Reverting to oracle labels...")

                # Calculate confidence differences
                confidence_diffs = {}
                for idx, meta in self.prev_labels.items():
                    if idx in self.previous_confidences:
                        # Find the current confidence for this sample
                        current_conf = current_labels[idx]['confidence'] if idx in current_labels else 0
                        confidence_diffs[idx] = abs(current_conf - self.previous_confidences[idx])

                # Select samples with the largest confidence differences
                sorted_diffs = sorted(confidence_diffs.items(), key=lambda item: item[1], reverse=True)
                oracle_correction_indices = [idx for idx, diff in sorted_diffs[:self.n_to_query]] #self.oracle_correction_size
                self.n_iteration_from_quering = 1
                print(f"Selected {len(oracle_correction_indices)} samples for oracle correction.")
                corrected_count = 0
                for idx in oracle_correction_indices:
                    if self.used_oracle_budget >= self.max_oracle_budget:
                        print("Oracle budget exceeded.")
                        break
                    if idx in self.prev_labels: # Ensure the index is from the previous iteration's self-labeled samples
                        self.dataset_manager.pseudo_labels_dict[idx] = self.prev_labels[idx]['oracle_label']
                        self.used_oracle_budget += 1
                        corrected_count += 1
                        self.prev_labels.pop(idx) # Remove from previous labels after correction

                self.current_oracle = corrected_count
                self.prev_labels = {}  # Clear previous labels after potential correction
            else:
                print("Accuracy dropped, but oracle correction is disabled (pure self-labeling).")
        else:
            self.n_iteration_from_quering +=1

            # Save current iteration’s labels and confidences 
            self.prev_labels = current_labels
            self.previous_confidences = {idx: meta['confidence'] for idx, meta in current_labels.items()}
            self.current_oracle = 0

        self.prev_accuracy = test_accuracy

class ActiveLearningLoop:
    def __init__(self, model_wrapper, dataset_manager, acquisition_size, strategy):
        self.model_wrapper = model_wrapper
        self.dataset_manager = dataset_manager
        self.acquisition_size = acquisition_size
        self.strategy = strategy.lower()

        self.current_oracle = acquisition_size
        self.used_oracle_budget = len(dataset_manager.labeled_indices) + len(dataset_manager.val_indices)

    def run_iteration(self):
        _, _, unlabeled_loader, _ = self.dataset_manager.get_loaders()
        unlabeled_indices = self.dataset_manager.unlabeled_indices

        if self.strategy == "entropy":
            entropies = self.model_wrapper.predict_entropies(unlabeled_loader)
            topk_indices_relative = np.argsort(entropies)[-self.acquisition_size:]
            print(f"Selected {len(topk_indices_relative)} most uncertain samples via entropy")

        elif self.strategy == "random":
            topk_indices_relative = np.random.choice(
                len(unlabeled_indices), size=self.acquisition_size, replace=False
            )
            print(f"Selected {len(topk_indices_relative)} random samples")

        else:
            raise ValueError(f"Unknown acquisition strategy: {self.strategy}")

        # Convert relative indices to absolute dataset indices
        topk_indices = np.array(unlabeled_indices)[topk_indices_relative]

        # Update dataset indices
        self.dataset_manager.update_indices(topk_indices)
        self.used_oracle_budget += self.acquisition_size



class HybridActiveSelfLabelingLoop:
    def __init__(self, model_wrapper, dataset_manager,
                 k_oracle_per_iter=200, k_selflabel_per_iter=200,
                 confidence_threshold=0.95, max_oracle_budget=2500,
                 oracle_correction_size=200, use_oracle_correction=True):

        self.model_wrapper = model_wrapper
        self.dataset_manager = dataset_manager
        self.k_oracle_per_iter = k_oracle_per_iter
        self.k_selflabel_per_iter = k_selflabel_per_iter
        self.confidence_threshold = confidence_threshold
        self.max_oracle_budget = max_oracle_budget

        # Attributes for tracking state across iterations for correction
        self.prev_selflabeled_samples = {}
        self.prev_accuracy = 0 

        # Attributes for logging and budget
        self.current_oracle_in_iter = 0 
        self.current_selflabeled_in_iter = 0 
        self.used_oracle_budget = 0 
        self.oracle_correction_size = oracle_correction_size
        self.use_oracle_correction = use_oracle_correction
        self.previous_confidences_for_correction = {} 

    def run_iteration(self, val_loader):
        # Reset counts for the current iteration
        self.current_oracle_in_iter = 0
        self.current_selflabeled_in_iter = 0
        samples_added_this_iter = []

        _, _, unlabeled_loader, _ = self.dataset_manager.get_loaders()

        entropies, confidences, predictions = self.model_wrapper.predict_entropy_and_confidence(unlabeled_loader)

        # Check for accuracy drop and perform correction if enabled
        current_accuracy = self.model_wrapper.evaluate(val_loader)
        print(f"Validation Accuracy (Current): {current_accuracy:.2f}%")

        if self.use_oracle_correction and current_accuracy < self.prev_accuracy and self.prev_selflabeled_samples:
            print("Accuracy dropped! Attempting to correct previously self-labeled samples using oracle.")

            # Calculate confidence differences for samples from the previous iteration
            confidence_diffs = {}
            for dataset_idx, meta in self.prev_selflabeled_samples.items():
                if dataset_idx in self.dataset_manager.unlabeled_indices:
                    local_unlabeled_idx = np.where(self.dataset_manager.unlabeled_indices == dataset_idx)[0][0]
                    current_conf_for_correction = confidences[local_unlabeled_idx]
                    confidence_diffs[dataset_idx] = abs(current_conf_for_correction - meta['confidence_when_selflabeled'])

            # Select samples with the largest confidence differences for correction
            sorted_diffs = sorted(confidence_diffs.items(), key=lambda item: item[1], reverse=True)
            oracle_correction_candidates = [idx for idx, diff in sorted_diffs] 

            corrected_count = 0
            for dataset_idx in oracle_correction_candidates:
                if corrected_count >= self.oracle_correction_size or self.used_oracle_budget >= self.max_oracle_budget:
                    print(f"Oracle correction limit reached ({corrected_count}/{self.oracle_correction_size}) or budget exceeded ({self.used_oracle_budget}/{self.max_oracle_budget}).")
                    break

                if dataset_idx in self.prev_selflabeled_samples:
                    # Replace the pseudo label with the true oracle label
                    _, true_label = self.dataset_manager.trainset[dataset_idx]
                    self.dataset_manager.pseudo_labels_dict[dataset_idx] = true_label
                    self.used_oracle_budget += 1
                    self.current_oracle_in_iter += 1
                    corrected_count += 1
                    # Remove from previous self-labeled samples as it's now oracle-labeled
                    del self.prev_selflabeled_samples[dataset_idx]


            print(f"Corrected {corrected_count} samples using oracle.")
            self.prev_accuracy = current_accuracy # Update previous accuracy for the next iteration's check
            return 

        # If no accuracy drop or correction is disabled, proceed with selection and labeling
        num_oracle_to_query = min(self.k_oracle_per_iter,
                                  len(self.dataset_manager.unlabeled_indices),
                                  self.max_oracle_budget - self.used_oracle_budget)
        oracle_local_indices = np.argsort(entropies)[-num_oracle_to_query:]
        print(f"Querying {len(oracle_local_indices)} samples for oracle labeling.")

        # Select for Self-Labeling and exclude samples already selected for oracle labeling from self-labeling candidates
        confident_local_indices_all = np.argsort(confidences)[::-1]  # Descending
        confident_local_indices_above_threshold = [
            i for i in confident_local_indices_all
            if confidences[i] >= self.confidence_threshold and i not in oracle_local_indices
        ]

        # Limit number of self-labeled points
        num_selflabel_to_add = min(self.k_selflabel_per_iter, len(confident_local_indices_above_threshold))
        selflabel_local_indices = confident_local_indices_above_threshold[:num_selflabel_to_add]
        print(f"Self-labeling {len(selflabel_local_indices)} samples.")

        # Get original dataset indices for selected samples
        oracle_dataset_indices = self.dataset_manager.unlabeled_indices[oracle_local_indices]
        selflabel_dataset_indices = self.dataset_manager.unlabeled_indices[selflabel_local_indices]

        # Apply oracle labels (true labels from the dataset)
        for local_i in oracle_local_indices:
            dataset_idx = self.dataset_manager.unlabeled_indices[local_i]
            _, true_label = self.dataset_manager.trainset[dataset_idx]
            self.dataset_manager.pseudo_labels_dict[dataset_idx] = true_label
            self.used_oracle_budget += 1
            self.current_oracle_in_iter += 1
            samples_added_this_iter.append(dataset_idx) 

        # Apply SL
        current_selflabeled_info = {}
        for local_i in selflabel_local_indices:
            dataset_idx = self.dataset_manager.unlabeled_indices[local_i]
            predicted_label = predictions[local_i]
            confidence_when_selflabeled = confidences[local_i]

            self.dataset_manager.pseudo_labels_dict[dataset_idx] = predicted_label
            self.current_selflabeled_in_iter += 1
            samples_added_this_iter.append(dataset_idx) 

            # Store info for potential correction
            current_selflabeled_info[dataset_idx] = {
                'pseudo_label': predicted_label,
                'oracle_label': self.dataset_manager.trainset[dataset_idx][1], # Store true label for correction
                'confidence_when_selflabeled': confidence_when_selflabeled
            }


        # Update the labeled and unlabeled pools
        all_added_local_indices = np.concatenate([oracle_local_indices, selflabel_local_indices])
        self.dataset_manager.labeled_indices = np.concatenate([
            self.dataset_manager.labeled_indices,
            self.dataset_manager.unlabeled_indices[all_added_local_indices]
        ])
        self.dataset_manager.unlabeled_indices = np.delete(self.dataset_manager.unlabeled_indices, all_added_local_indices)


        # Update previous SL samples and accuracy for the next iteration
        self.prev_selflabeled_samples = current_selflabeled_info
        self.prev_accuracy = current_accuracy 

        print(f"Oracle Budget Used (Total): {self.used_oracle_budget}/{self.max_oracle_budget}")

    def run_iteration(self, val_loader):
        self.current_oracle_in_iter = 0
        self.current_selflabeled_in_iter = 0
        samples_added_this_iter = []
        correction_applied = False

        _, _, unlabeled_loader, _ = self.dataset_manager.get_loaders()
        entropies, confidences, predictions = self.model_wrapper.predict_entropy_and_confidence(unlabeled_loader)
        current_accuracy = self.model_wrapper.evaluate(val_loader)
        print(f"Validation Accuracy (Current): {current_accuracy:.2f}%")

        # Oracle Correction 
        if self.use_oracle_correction and current_accuracy < self.prev_accuracy and self.prev_selflabeled_samples:
            print("Accuracy dropped — triggering oracle correction...")
            confidence_diffs = {}

            # Consider samples that still in the unlabeled pool
            self.prev_selflabeled_samples = {
                k: v for k, v in self.prev_selflabeled_samples.items()
                if k in self.dataset_manager.unlabeled_indices
            }

            for dataset_idx, meta in self.prev_selflabeled_samples.items():
                local_idx = np.where(self.dataset_manager.unlabeled_indices == dataset_idx)[0]
                if len(local_idx) == 0:
                    continue
                current_conf = confidences[local_idx[0]]
                confidence_diffs[dataset_idx] = abs(current_conf - meta['confidence_when_selflabeled'])

            sorted_diffs = sorted(confidence_diffs.items(), key=lambda item: item[1], reverse=True)
            candidates = [idx for idx, _ in sorted_diffs]

            corrected = 0
            for dataset_idx in candidates:
                if corrected >= self.oracle_correction_size or self.used_oracle_budget >= self.max_oracle_budget:
                    break

                _, true_label = self.dataset_manager.trainset[dataset_idx]
                self.dataset_manager.pseudo_labels_dict[dataset_idx] = true_label
                self.dataset_manager.labeled_indices = np.append(self.dataset_manager.labeled_indices, dataset_idx)
                self.dataset_manager.unlabeled_indices = np.setdiff1d(self.dataset_manager.unlabeled_indices, [dataset_idx])
                self.used_oracle_budget += 1
                self.current_oracle_in_iter += 1
                corrected += 1

                if dataset_idx in self.prev_selflabeled_samples:
                    del self.prev_selflabeled_samples[dataset_idx]

            print(f"Corrected {corrected} samples via oracle.")
            self.prev_accuracy = current_accuracy
            return True  

        # Oracle+SL
        num_oracle_to_query = min(self.k_oracle_per_iter,
                                  len(self.dataset_manager.unlabeled_indices),
                                  self.max_oracle_budget - self.used_oracle_budget)
        oracle_local_indices = np.argsort(entropies)[-num_oracle_to_query:]
        confident_local_indices_all = np.argsort(confidences)[::-1]
        confident_local_indices_above_threshold = [
            i for i in confident_local_indices_all
            if confidences[i] >= self.confidence_threshold and i not in oracle_local_indices
        ]
        num_selflabel_to_add = min(self.k_selflabel_per_iter, len(confident_local_indices_above_threshold))
        selflabel_local_indices = confident_local_indices_above_threshold[:num_selflabel_to_add]

        oracle_dataset_indices = self.dataset_manager.unlabeled_indices[oracle_local_indices]
        selflabel_dataset_indices = self.dataset_manager.unlabeled_indices[selflabel_local_indices]

        # Add oracle labels
        for local_i in oracle_local_indices:
            dataset_idx = self.dataset_manager.unlabeled_indices[local_i]
            _, true_label = self.dataset_manager.trainset[dataset_idx]
            self.dataset_manager.pseudo_labels_dict[dataset_idx] = true_label
            self.used_oracle_budget += 1
            self.current_oracle_in_iter += 1
            samples_added_this_iter.append(dataset_idx)

        # Add SL
        current_selflabeled_info = {}
        for local_i in selflabel_local_indices:
            dataset_idx = self.dataset_manager.unlabeled_indices[local_i]
            predicted_label = predictions[local_i]
            confidence = confidences[local_i]
            self.dataset_manager.pseudo_labels_dict[dataset_idx] = predicted_label
            self.current_selflabeled_in_iter += 1
            samples_added_this_iter.append(dataset_idx)
            current_selflabeled_info[dataset_idx] = {
                'pseudo_label': predicted_label,
                'oracle_label': self.dataset_manager.trainset[dataset_idx][1],
                'confidence_when_selflabeled': confidence
            }

        # Update DU and DL
        all_added_local_indices = np.concatenate([oracle_local_indices, selflabel_local_indices])
        self.dataset_manager.labeled_indices = np.concatenate([
            self.dataset_manager.labeled_indices,
            self.dataset_manager.unlabeled_indices[all_added_local_indices]
        ])
        self.dataset_manager.unlabeled_indices = np.delete(self.dataset_manager.unlabeled_indices, all_added_local_indices)

        self.prev_selflabeled_samples = current_selflabeled_info
        self.prev_accuracy = current_accuracy

        print(f"Oracle Budget Used : {self.used_oracle_budget}/{self.max_oracle_budget}")
        return False  

