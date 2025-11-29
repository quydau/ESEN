# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.esen import ESEN, compute_loss
from utils.preprocessing import ESENDataPreprocessor
from models.graph_constructor import GraphConstructor
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Detect device (CUDA hoặc MPS cho Mac)
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # Mac M1/M2
        return torch.device('mps')
    else:
        return torch.device('cpu')

def move_batch_to_device(batch, device):
    """Recursively move batch tensors to device"""
    
    def move_dict(d, device):
        result = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, dict):
                result[k] = move_dict(v, device)
            elif isinstance(v, list):
                result[k] = [move_item(item, device) for item in v]
            else:
                result[k] = v
        return result
    
    def move_item(item, device):
        if isinstance(item, torch.Tensor):
            return item.to(device)
        elif isinstance(item, dict):
            return move_dict(item, device)
        elif isinstance(item, list):
            return [move_item(i, device) for i in item]
        return item
    
    return move_dict(batch, device)

class FakeNewsDataset(Dataset):
    """Dataset cho PolitiFact fake news detection"""
    
    def __init__(self, data_file: str, graph_constructor: GraphConstructor, 
                 preprocessor: ESENDataPreprocessor):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.keys = list(self.data.keys())
        self.graph_constructor = graph_constructor
        self.preprocessor = preprocessor
        
        # Label mapping
        self.true_labels = {'true'}
        self.false_labels = {'false'}
    
    def __len__(self):
        return len(self.keys)
    
    def _get_binary_label(self, cred_label: str) -> int:
        """Convert 6-class label to binary"""
        label_lower = cred_label.lower()
        if label_lower in self.true_labels:
            return 1  # True
        elif label_lower in self.false_labels:
            return 0  # Fake
        else:
            raise ValueError(f"Unknown label: {cred_label}")
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]
        
        # Build graphs
        graphs = self.graph_constructor.process_claim_evidence(
            claim=sample['claim_text'],
            evidences=sample['evidences']
        )
        
        # Prepare tensors
        prepared = self.preprocessor.prepare_sample(graphs)
        
        # Binary label
        label = self._get_binary_label(sample['cred_label'])
        
        # Publisher embeddings (TODO: implement real publisher encoder)
        P_c = torch.randn(100)
        P_e_list = [torch.randn(100) for _ in sample['evidences']]
        
        # Adjacency matrices
        adj_e_list = [evi['semantic']['adj_matrix'] for evi in graphs['evidences']]
        
        return {
            'prepared_sample': prepared,
            'P_c': P_c,
            'P_e_list': P_e_list,
            'adj_e_list': adj_e_list,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        batch = move_batch_to_device(batch, device)
        label = batch['label']
        
        # Forward
        logits = model(batch)
        
        # Loss
        loss = compute_loss(logits, label)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        pred = torch.argmax(logits)
        correct += (pred == label).item()
        total += 1
    
    return total_loss / total, correct / total


def evaluate(model, dataloader, device):
    """Evaluate với classification report"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            label = batch['label']
            logits = model(batch)
            loss = compute_loss(logits, label)
            
            total_loss += loss.item()
            pred = torch.argmax(logits).cpu().item()
            
            all_preds.append(pred)
            all_labels.append(label.cpu().item())
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=['Fake', 'True'],
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, report, cm


def collate_fn(batch):
    """
    Custom collate vì mỗi sample có số evidence/nodes khác nhau.
    Không thể stack thành batch → xử lý từng sample.
    """
    if len(batch) == 1:
        return batch[0]
    
    # Nếu muốn batch > 1, cần pad/truncate (phức tạp)
    # Đơn giản: return list
    return batch


def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Results storage
    fold_results = []
    
    for fold in range(5):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        # Data
        graph_constructor = GraphConstructor(window_size=3)
        preprocessor = ESENDataPreprocessor(embedding_dim=300)
        
        train_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/train_{fold}.json',
            graph_constructor, preprocessor
        )
        test_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/test_{fold}.json',
            graph_constructor, preprocessor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Model
        model = ESEN(input_dim=300, hidden_dim=300, publisher_dim=100).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        
        # Train
        best_acc = 0
        for epoch in range(50):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc, test_report, test_cm = evaluate(model, test_loader, device)
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}:")
                print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
                print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_report = test_report
                best_cm = test_cm
                torch.save(model.state_dict(), f'checkpoints/esen_fold{fold}.pth')
        
        # Final results
        print(f"\n{'='*60}")
        print(f"Fold {fold} Best Results (Acc: {best_acc:.4f})")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(best_report)
        print("\nConfusion Matrix:")
        print(best_cm)
        
        fold_results.append({
            'fold': fold,
            'accuracy': best_acc,
            'report': best_report,
            'cm': best_cm
        })
    
    # Average across folds
    print(f"\n{'='*60}")
    print("5-Fold Cross Validation Results")
    print(f"{'='*60}")
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    print(f"Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
    for r in fold_results:
        print(f"\nFold {r['fold']}:")
        print(r['report'])


if __name__ == "__main__":
    import os
    os.makedirs('checkpoints', exist_ok=True)
    main()
