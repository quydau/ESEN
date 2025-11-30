# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.esen import ESEN, compute_loss
from utils.preprocessing import ESENDataPreprocessor
from models.graph_constructor import GraphConstructor
import json
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from multiprocessing import Pool, cpu_count

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
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

def _process_sample(args):
    """Helper function for parallel preprocessing"""
    key, sample, window_size, stanza_cache_dir = args
    
    # Create instances for each worker
    graph_constructor = GraphConstructor(window_size=window_size, cache_dir=stanza_cache_dir)
    preprocessor = ESENDataPreprocessor(embedding_dim=300)
    
    graphs = graph_constructor.process_claim_evidence(
        claim=sample['claim_text'],
        evidences=sample['evidences']
    )
    
    prepared = preprocessor.prepare_sample(graphs)
    
    # Binary label
    label_lower = sample['cred_label'].lower()
    if label_lower in {'true'}:
        label = 1
    elif label_lower in {'false'}:
        label = 0
    else:
        raise ValueError(f"Unknown label: {sample['cred_label']}")
    
    adj_e_list = [evi['semantic']['adj_matrix'] for evi in graphs['evidences']]
    
    return {
        'prepared_sample': prepared,
        'adj_e_list': adj_e_list,
        'label': label
    }

class FakeNewsDataset(Dataset):
    """Dataset with parallel preprocessing and caching"""
    
    def __init__(self, data_file: str, window_size: int = 3, 
                 cache_dir: str = 'preprocessed_cache',
                 stanza_cache_dir: str = 'stanza_cache',
                 n_workers: int = None):
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(stanza_cache_dir, exist_ok=True)
        
        # Cache path
        cache_filename = os.path.basename(data_file).replace('.json', '_preprocessed.pkl')
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # Load from cache if exists
        if os.path.exists(self.cache_path):
            print(f"Loading preprocessed data from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.samples = cached_data['samples']
                self.keys = cached_data['keys']
            print(f"✓ Loaded {len(self.samples)} preprocessed samples")
        else:
            # Load raw data
            with open(data_file, 'r') as f:
                self.data = json.load(f)
            
            self.keys = list(self.data.keys())
            
            # Determine number of workers
            if n_workers is None:
                n_workers = min(cpu_count() - 1, 8)
            
            print(f"Preprocessing {len(self.keys)} samples with {n_workers} workers...")
            
            # Prepare arguments for parallel processing
            args_list = [
                (key, self.data[key], window_size, stanza_cache_dir)
                for key in self.keys
            ]
            
            # Parallel preprocessing
            if n_workers > 1:
                with Pool(n_workers) as pool:
                    self.samples = list(tqdm(
                        pool.imap(_process_sample, args_list),
                        total=len(args_list),
                        desc="Preprocessing"
                    ))
            else:
                # Sequential fallback
                self.samples = [_process_sample(args) for args in tqdm(args_list, desc="Preprocessing")]
            
            # Save cache
            print(f"Saving preprocessed data to {self.cache_path}...")
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'samples': self.samples, 'keys': self.keys}, f)
            print("✓ Cache saved successfully")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Proper batching collate function"""
    return {
        'prepared_sample': [b['prepared_sample'] for b in batch],
        'adj_e_list': [b['adj_e_list'] for b in batch],
        'label': torch.stack([torch.tensor(b['label'], dtype=torch.long) for b in batch])
    }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = move_batch_to_device(batch, device)
        labels = batch['label']
        
        # Forward pass for batch
        logits_list = []
        for i in range(len(batch['prepared_sample'])):
            single_batch = {
                'prepared_sample': batch['prepared_sample'][i],
                'adj_e_list': batch['adj_e_list'][i],
                'label': labels[i]
            }
            logits = model(single_batch)
            logits_list.append(logits)
        
        # Stack logits and compute loss
        logits_batch = torch.stack(logits_list)
        loss = compute_loss(logits_batch, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits_batch, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    return avg_loss, accuracy, f1_macro, f1_micro

def evaluate(model, dataloader, device):
    """Evaluate with classification report and F1 scores"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            labels = batch['label']
            
            # Forward pass for batch
            logits_list = []
            for i in range(len(batch['prepared_sample'])):
                single_batch = {
                    'prepared_sample': batch['prepared_sample'][i],
                    'adj_e_list': batch['adj_e_list'][i],
                    'label': labels[i]
                }
                logits = model(single_batch)
                logits_list.append(logits)
            
            logits_batch = torch.stack(logits_list)
            loss = compute_loss(logits_batch, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits_batch, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
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
    
    cm = confusion_matrix(all_labels, all_preds)
    
    avg_loss = total_loss / len(dataloader)
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

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Early stopping config
    patience = 10
    min_delta = 0.0001
    
    # Preprocessing workers
    n_workers = 8
    
    fold_results = []
    
    for fold in range(1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        # Load datasets with parallel preprocessing and caching
        train_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/train_{fold}.json',
            window_size=3,
            cache_dir='preprocessed_cache',
            stanza_cache_dir='stanza_cache',
            n_workers=n_workers
        )
        dev_dataset = FakeNewsDataset(
            'data/PolitiFact/json/dev.json',
            window_size=3,
            cache_dir='preprocessed_cache',
            stanza_cache_dir='stanza_cache',
            n_workers=n_workers
        )
        test_dataset = FakeNewsDataset(
            f'data/PolitiFact/json/5fold/test_{fold}.json',
            window_size=3,
            cache_dir='preprocessed_cache',
            stanza_cache_dir='stanza_cache',
            n_workers=n_workers
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=64, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
        
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
        best_dev_f1 = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(50):
            # Train
            train_loss, train_acc, train_f1_macro, train_f1_micro = train_epoch(
                model, train_loader, optimizer, device
            )
            
            # Evaluate on dev
            dev_results = evaluate(model, dev_loader, device)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}:")
                print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1-Ma: {train_f1_macro:.4f}")
                print(f"Dev   - Loss: {dev_results['loss']:.4f} | Acc: {dev_results['accuracy']:.4f} | F1-Ma: {dev_results['f1_macro']:.4f}")
            
            # Early stopping check
            current_dev_f1 = dev_results['f1_macro']
            
            if current_dev_f1 > best_dev_f1 + min_delta:
                best_dev_f1 = current_dev_f1
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_f1': current_dev_f1,
                }, f'checkpoints/esen_fold{fold}_best.pth')
                
                print(f"✓ New best Dev F1-Macro: {best_dev_f1:.4f} at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best Dev F1-Macro: {best_dev_f1:.4f} at epoch {best_epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(f'checkpoints/esen_fold{fold}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        final_test_results = evaluate(model, test_loader, device)
        
        print(f"\n{'='*60}")
        print(f"Fold {fold} - Best Results (Dev F1-Macro: {best_dev_f1:.4f})")
        print(f"{'='*60}")
        print(f"\nTest Performance:")
        print(f"Accuracy:  {final_test_results['accuracy']:.4f}")
        print(f"F1-Macro:  {final_test_results['f1_macro']:.4f}")
        print(f"F1-Micro:  {final_test_results['f1_micro']:.4f}")
        print(f"\n{final_test_results['report_str']}")
        print(f"\nConfusion Matrix:\n{final_test_results['cm']}")
        
        fold_results.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'dev_f1_macro': best_dev_f1,
            'test_accuracy': final_test_results['accuracy'],
            'test_f1_macro': final_test_results['f1_macro'],
            'test_f1_micro': final_test_results['f1_micro'],
            'report': final_test_results['report'],
            'cm': final_test_results['cm']
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("5-Fold Cross Validation Results")
    print(f"{'='*60}")
    
    avg_acc = np.mean([r['test_accuracy'] for r in fold_results])
    std_acc = np.std([r['test_accuracy'] for r in fold_results])
    avg_f1_ma = np.mean([r['test_f1_macro'] for r in fold_results])
    std_f1_ma = np.std([r['test_f1_macro'] for r in fold_results])
    avg_f1_mi = np.mean([r['test_f1_micro'] for r in fold_results])
    std_f1_mi = np.std([r['test_f1_micro'] for r in fold_results])
    
    print(f"Test Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Test F1-Macro:  {avg_f1_ma:.4f} ± {std_f1_ma:.4f}")
    print(f"Test F1-Micro:  {avg_f1_mi:.4f} ± {std_f1_mi:.4f}")

if __name__ == "__main__":
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('preprocessed_cache', exist_ok=True)
    os.makedirs('stanza_cache', exist_ok=True)
    main()
