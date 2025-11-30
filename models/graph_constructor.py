import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import stanza
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import os


class GraphConstructor:
    """Xây dựng đồ thị ngữ nghĩa và cú pháp với cache và parallel processing"""
    
    def __init__(self, window_size: int = 3, cache_dir: str = 'cache'):
        self.window_size = window_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Khởi tạo Stanza một lần
        self.nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            use_gpu=torch.cuda.is_available(),
            download_method=None  # Không download lại nếu đã có
        )
    
    def _get_cache_path(self, text: str) -> str:
        """Tạo cache path từ hash của text"""
        text_hash = hash(text) % (10 ** 8)
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def _load_from_cache(self, text: str):
        """Load từ cache nếu có"""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, text: str, data):
        """Lưu vào cache"""
        cache_path = self._get_cache_path(text)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def parse_text(self, text: str):
        """Parse text một lần, cache kết quả"""
        cached = self._load_from_cache(text)
        if cached is not None:
            return cached
        
        doc = self.nlp(text)
        
        # Extract words và dependencies
        words = []
        dependencies = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text.lower())
            
            for dep in sentence.dependencies:
                head_idx = dep[0].id - 1
                dep_idx = dep[2].id - 1
                dep_relation = dep[1]
                
                if head_idx >= 0:
                    dependencies.append((head_idx, dep_idx, dep_relation))
        
        result = {'words': words, 'dependencies': dependencies}
        self._save_to_cache(text, result)
        return result
    
    def build_semantic_graph(self, parsed_data: Dict) -> Tuple[torch.Tensor, List[str], Dict]:
        """Xây dựng semantic graph từ parsed data"""
        words = parsed_data['words']
        
        # Merge từ lặp
        unique_words = []
        word2idx = {}
        idx = 0
        
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                unique_words.append(word)
                idx += 1
        
        n_nodes = len(unique_words)
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Sliding window - vectorized
        for i, word in enumerate(words):
            node_i = word2idx[word]
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    node_j = word2idx[words[j]]
                    adj_matrix[node_i, node_j] = 1.0
        
        # Self-loop
        adj_matrix += np.eye(n_nodes)
        
        return torch.FloatTensor(adj_matrix), unique_words, word2idx
    
    def build_syntactic_graph(self, parsed_data: Dict) -> Tuple[torch.Tensor, List[str], Dict]:
        """Xây dựng syntactic graph từ parsed data"""
        words = parsed_data['words']
        dependencies = parsed_data['dependencies']
        
        # Merge từ lặp
        unique_words = []
        word2idx = {}
        original2new = {}
        idx = 0
        
        for orig_idx, word in enumerate(words):
            if word not in word2idx:
                word2idx[word] = idx
                unique_words.append(word)
                original2new[orig_idx] = idx
                idx += 1
            else:
                original2new[orig_idx] = word2idx[word]
        
        n_nodes = len(unique_words)
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Tính trọng số
        dep_weights = defaultdict(float)
        
        for head_orig, dep_orig, relation in dependencies:
            head_new = original2new[head_orig]
            dep_new = original2new[dep_orig]
            weight = self._calculate_syntactic_weight(relation)
            
            dep_weights[(head_new, dep_new)] += weight
            dep_weights[(dep_new, head_new)] += weight
        
        for (i, j), weight in dep_weights.items():
            adj_matrix[i, j] = weight
        
        adj_matrix += np.eye(n_nodes)
        
        return torch.FloatTensor(adj_matrix), unique_words, word2idx
    
    def _calculate_syntactic_weight(self, relation: str) -> float:
        important_deps = {
            'nsubj': 1.0, 'obj': 1.0, 'iobj': 0.9, 'csubj': 0.9,
            'amod': 0.8, 'advmod': 0.7, 'compound': 0.8, 'conj': 0.6,
        }
        return important_deps.get(relation, 0.5)
    
    def process_claim_evidence(self, claim: str, evidences: List[List]) -> Dict:
        """Xử lý claim và evidences - parse một lần, dùng lại"""
        # Parse claim
        claim_parsed = self.parse_text(claim)
        claim_sem_adj, claim_sem_vocab, claim_sem_w2i = self.build_semantic_graph(claim_parsed)
        claim_syn_adj, claim_syn_vocab, claim_syn_w2i = self.build_syntactic_graph(claim_parsed)
        
        # Parse evidences
        evidences_graphs = []
        for evi_id, evi_text, evi_source in evidences:
            evi_parsed = self.parse_text(evi_text)
            evi_sem_adj, evi_sem_vocab, evi_sem_w2i = self.build_semantic_graph(evi_parsed)
            evi_syn_adj, evi_syn_vocab, evi_syn_w2i = self.build_syntactic_graph(evi_parsed)
            
            evidences_graphs.append({
                'id': evi_id,
                'source': evi_source,
                'semantic': {
                    'adj_matrix': evi_sem_adj,
                    'vocab': evi_sem_vocab,
                    'word2idx': evi_sem_w2i
                },
                'syntactic': {
                    'adj_matrix': evi_syn_adj,
                    'vocab': evi_syn_vocab,
                    'word2idx': evi_syn_w2i
                }
            })
        
        return {
            'claim': {
                'text': claim,
                'semantic': {
                    'adj_matrix': claim_sem_adj,
                    'vocab': claim_sem_vocab,
                    'word2idx': claim_sem_w2i
                },
                'syntactic': {
                    'adj_matrix': claim_syn_adj,
                    'vocab': claim_syn_vocab,
                    'word2idx': claim_syn_w2i
                }
            },
            'evidences': evidences_graphs
        }


# Parallel preprocessing helper
def process_single_sample(args):
    """Helper cho multiprocessing"""
    key, sample, window_size, cache_dir = args
    graph_constructor = GraphConstructor(window_size=window_size, cache_dir=cache_dir)
    graphs = graph_constructor.process_claim_evidence(
        claim=sample['claim_text'],
        evidences=sample['evidences']
    )
    return key, graphs


def preprocess_dataset_parallel(data_file: str, output_file: str, 
                                window_size: int = 3, n_workers: int = None):
    """Preprocessing song song toàn bộ dataset"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if n_workers is None:
        n_workers = min(cpu_count() - 1, 8)  # Tối đa 8 workers
    
    print(f"Processing {len(data)} samples with {n_workers} workers...")
    
    # Chuẩn bị args
    args_list = [
        (key, sample, window_size, 'cache')
        for key, sample in data.items()
    ]
    
    # Parallel processing
    with Pool(n_workers) as pool:
        results = pool.map(process_single_sample, args_list)
    
    # Lưu kết quả
    processed_data = dict(results)
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved preprocessed data to {output_file}")


if __name__ == "__main__":
    # Preprocessing song song
    preprocess_dataset_parallel(
        'data/PolitiFact/json/5fold/train_0.json',
        'data/PolitiFact/preprocessed/train_0.pkl',
        n_workers=8
    )
