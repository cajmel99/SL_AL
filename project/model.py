import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class SemiSupervisedModel:
    def __init__(self, num_classes=10, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_dataloader, val_dataloader, epochs=5, patience=10):
        self.model.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Evaluate on validation set
            val_loss = self.evaluate_loss(val_dataloader)
            print(f"Epoch {epoch+1}: Training Loss = {running_loss / len(train_dataloader):.4f}, Validation Loss = {val_loss:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        return acc

    def evaluate_loss(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)


    def predict_confidences(self, dataloader):
        self.model.eval()
        all_confs, all_preds = [], []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)
                all_confs.extend(confs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Convert to numpy arrays to ensure correct shape handling
        all_confs = np.array(all_confs)
        all_preds = np.array(all_preds)

        return all_confs, all_preds


    def predict_entropies(self, dataloader):
        self.model.eval()
        all_entropies = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # Avoid log(0)
                all_entropies.extend(entropy.cpu().numpy())
        return np.array(all_entropies)

    def predict_entropy_and_confidence(self, dataloader):
        self.model.eval()
        entropies = []
        confidences = []
        predictions = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                confidence, preds = torch.max(probs, dim=1)

                entropies.extend(entropy.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

        return np.array(entropies), np.array(confidences), np.array(predictions)