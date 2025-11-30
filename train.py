# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.esen import ESEN, compute_loss
from utils.preprocessing import ESENDataPreprocessor
from models.graph_constructor import GraphConstructor
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
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
                 preprocessor: ESENDataPreprocessor, keys_subset=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        # Allow subset of keys for train/val split
        self.keys = keys_subset if keys_subset is not None else list(self.data.keys())
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
    all_preds = []
    all_labels = []
    
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
        pred = torch.argmax(logits).cpu().item()
        all_preds.append(pred)
        all_labels.append(label.cpu().item())
    
    # Calculate metrics
    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    return avg_loss, accuracy, f1_macro, f1_micro


def evaluate(model, dataloader, device):
    """Evaluate với classification report và F1 scores"""
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
        digits=4,
        output_dict=True
    )
    
    report_str = classification_report(
        all_labels, all_preds,
        target_names=['Fake', 'True'],
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Metrics
    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'report': report,
        'report_str': report_str,
        'cm': cm
    }


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
    
    # Early stopping config
    patience = 10
    min_delta = 0.0001  # Minimum improvement to consider
    
    # Results storage
    fold_results = []
    
    for fold in range(1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        # Data
        graph_constructor = GraphConstructor(window_size=3)
        preprocessor = ESENDataPreprocessor(embedding_dim=300)
        
        # Load training data and split into train/val (90/10)
        with open(f'data/PolitiFact/json/5fold/train_{fold}.json', 'r') as f:
            train_data = json.load(f)
        
        all_keys = list(train_data.keys())
        train_keys, val_keys = train_test_split(
            all_keys, 
            test_size=0.1, 
            random_state=42,
            stratify=[train_data[k]['cred_label'].lower() for k in all_keys]
        )
        
        train_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/train_{fold}.json',
            graph_constructor, preprocessor,
            keys_subset=train_keys
        )
        val_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/train_{fold}.json',
            graph_constructor, preprocessor,
            keys_subset=val_keys
        )
        test_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/test_{fold}.json',
            graph_constructor, preprocessor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        
        # Model
        model = ESEN(
            input_dim=300,
            hidden_dim=300,
            publisher_dim=100,
            num_semantic_steps=5,
            num_attention_heads=4,
            num_dense_sublayers=3,
            num_evi_layers=2,
            alpha=0.5,
            top_k_ratio=0.3
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        
        # Training with early stopping
        best_val_f1 = 0
        best_epoch = 0
        patience_counter = 0
        best_test_results = None
        
        for epoch in range(50):
            # Train
            train_loss, train_acc, train_f1_macro, train_f1_micro = train_epoch(
                model, train_loader, optimizer, device
            )
            
            # Validate
            val_results = evaluate(model, val_loader, device)
            
            # Test (for tracking)
            test_results = evaluate(model, test_loader, device)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}:")
                print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1-Macro: {train_f1_macro:.4f}")
                print(f"Val   - Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | F1-Macro: {val_results['f1_macro']:.4f}")
                print(f"Test  - Loss: {test_results['loss']:.4f} | Acc: {test_results['accuracy']:.4f} | F1-Macro: {test_results['f1_macro']:.4f}")
            
            # Early stopping check based on validation F1-Macro
            current_val_f1 = val_results['f1_macro']
            
            if current_val_f1 > best_val_f1 + min_delta:
                best_val_f1 = current_val_f1
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': current_val_f1,
                }, f'checkpoints/esen_fold{fold}_best.pth')
                
                # Store best test results
                best_test_results = test_results
                
                print(f"✓ New best Val F1-Macro: {best_val_f1:.4f} at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Val F1-Macro: {best_val_f1:.4f} at epoch {best_epoch+1}")
                break
        
        # Load best model for final evaluation
        checkpoint = torch.load(f'checkpoints/esen_fold{fold}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        final_test_results = evaluate(model, test_loader, device)
        
        # Print final results
        print(f"\n{'='*60}")
        print(f"Fold {fold} - Best Results (Val F1-Macro: {best_val_f1:.4f})")
        print(f"{'='*60}")
        print(f"\nTest Performance:")
        print(f"Accuracy: {final_test_results['accuracy']:.4f}")
        print(f"F1-Macro: {final_test_results['f1_macro']:.4f}")
        print(f"F1-Micro: {final_test_results['f1_micro']:.4f}")
        print(f"\nClassification Report:")
        print(final_test_results['report_str'])
        print(f"\nConfusion Matrix:")
        print(final_test_results['cm'])
        
        fold_results.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'val_f1_macro': best_val_f1,
            'test_accuracy': final_test_results['accuracy'],
            'test_f1_macro': final_test_results['f1_macro'],
            'test_f1_micro': final_test_results['f1_micro'],
            'report': final_test_results['report'],
            'cm': final_test_results['cm']
        })
    
    # Average across folds
    print(f"\n{'='*60}")
    print("5-Fold Cross Validation Results")
    print(f"{'='*60}")
    
    avg_test_acc = np.mean([r['test_accuracy'] for r in fold_results])
    std_test_acc = np.std([r['test_accuracy'] for r in fold_results])
    
    avg_test_f1_macro = np.mean([r['test_f1_macro'] for r in fold_results])
    std_test_f1_macro = np.std([r['test_f1_macro'] for r in fold_results])
    
    avg_test_f1_micro = np.mean([r['test_f1_micro'] for r in fold_results])
    std_test_f1_micro = np.std([r['test_f1_micro'] for r in fold_results])
    
    print(f"Test Accuracy:  {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Test F1-Macro:  {avg_test_f1_macro:.4f} ± {std_test_f1_macro:.4f}")
    print(f"Test F1-Micro:  {avg_test_f1_micro:.4f} ± {std_test_f1_micro:.4f}")
    
    for r in fold_results:
        print(f"\nFold {r['fold']} (stopped at epoch {r['best_epoch']+1}):")
        print(f"Val F1-Macro: {r['val_f1_macro']:.4f}")
        print(f"Test F1-Macro: {r['test_f1_macro']:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs('checkpoints', exist_ok=True)
    main()
