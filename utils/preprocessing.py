# utils/preprocessing.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

class ESENDataPreprocessor:
    """
    Preprocessing tổng quát cho ESEN model
    """
    
    def __init__(self, 
                 vocab_file: str = None,
                 embedding_dim: int = 300,
                 pretrained_embeddings: str = 'glove'):
        """
        Args:
            vocab_file: Path đến vocabulary đã build
            embedding_dim: Dimension của word embeddings
            pretrained_embeddings: 'glove', 'word2vec', hoặc None (trainable)
        """
        self.embedding_dim = embedding_dim
        
        # Build hoặc load vocabulary
        if vocab_file and Path(vocab_file).exists():
            self.word2id, self.id2word = self._load_vocab(vocab_file)
        else:
            self.word2id = {'<PAD>': 0, '<UNK>': 1}
            self.id2word = {0: '<PAD>', 1: '<UNK>'}
        
        # Load pretrained embeddings
        self.embedding_matrix = self._init_embeddings(pretrained_embeddings)
    
    def _load_vocab(self, vocab_file: str) -> Tuple[Dict, Dict]:
        """Load vocabulary từ file"""
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        return vocab_data['word2id'], vocab_data['id2word']
    
    def _init_embeddings(self, pretrained: str) -> np.ndarray:
        """
        Khởi tạo embedding matrix
        
        Returns:
            embedding_matrix: [vocab_size, embedding_dim]
        """
        vocab_size = len(self.word2id)
        
        if pretrained == 'glove':
            # Load GloVe embeddings
            embedding_matrix = self._load_glove()
        elif pretrained == 'word2vec':
            # Load Word2Vec
            embedding_matrix = self._load_word2vec()
        else:
            # Random initialization (trainable)
            embedding_matrix = np.random.randn(vocab_size, self.embedding_dim) * 0.01
            embedding_matrix[0] = 0  # PAD = zeros
        
        return embedding_matrix
    
    def _load_glove(self) -> np.ndarray:
        """Load GloVe embeddings"""
        # Implementation: đọc file glove.6B.300d.txt
        # Map words trong vocab sang embeddings
        # Đơn giản hóa: return random matrix
        vocab_size = len(self.word2id)
        return np.random.randn(vocab_size, self.embedding_dim) * 0.01
    
    def build_vocab_from_graphs(self, all_graphs: List[Dict]):
        """
        Build vocabulary từ tất cả graphs đã tạo
        
        Args:
            all_graphs: List output từ GraphConstructor
        """
        word_freq = {}
        
        for graphs in all_graphs:
            # Claim vocab
            for word in graphs['claim']['semantic']['vocab']:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Evidence vocab
            for evi in graphs['evidences']:
                for word in evi['semantic']['vocab']:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add to vocabulary (frequency threshold có thể áp dụng)
        idx = len(self.word2id)
        for word in sorted(word_freq.keys()):
            if word not in self.word2id:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1
    
    def graph_to_tensors(self, graph_data: Dict) -> Dict:
        """
        Convert graph data sang tensors cho model
        
        Args:
            graph_data: Output từ GraphConstructor (semantic hoặc syntactic)
            
        Returns:
            {
                'node_features': [num_nodes, embedding_dim],
                'adj_matrix': [num_nodes, num_nodes],
                'mask': [num_nodes] (padding mask)
            }
        """
        vocab = graph_data['vocab']
        word2idx_local = graph_data['word2idx']
        adj_matrix = graph_data['adj_matrix']
        
        num_nodes = len(vocab)
        
        # Map local vocab indices -> global vocab indices
        node_ids = []
        for word in vocab:
            global_id = self.word2id.get(word, self.word2id['<UNK>'])
            node_ids.append(global_id)
        
        # Get embeddings for nodes
        node_features = torch.FloatTensor(
            self.embedding_matrix[node_ids]
        )  # [num_nodes, embedding_dim]
        
        # Adjacency matrix đã có từ GraphConstructor
        adj_matrix_tensor = graph_data['adj_matrix']  # Already tensor
        
        # Mask (all ones, no padding trong graph-level)
        mask = torch.ones(num_nodes, dtype=torch.bool)
        
        return {
            'node_features': node_features,
            'adj_matrix': adj_matrix_tensor,
            'mask': mask,
            'num_nodes': num_nodes
        }
    
    def prepare_sample(self, graphs: Dict) -> Dict:
        """
        Chuẩn bị một sample hoàn chỉnh cho ESEN
        
        Args:
            graphs: Output từ GraphConstructor.process_claim_evidence()
            
        Returns:
            Formatted sample với tensors
        """
        # Claim semantic
        claim_sem = self.graph_to_tensors(graphs['claim']['semantic'])
        # Claim syntactic  
        claim_syn = self.graph_to_tensors(graphs['claim']['syntactic'])
        
        # Evidences
        evidences_sem = []
        evidences_syn = []
        
        for evi in graphs['evidences']:
            evidences_sem.append(self.graph_to_tensors(evi['semantic']))
            evidences_syn.append(self.graph_to_tensors(evi['syntactic']))
        
        return {
            'claim': {
                'semantic': claim_sem,
                'syntactic': claim_syn
            },
            'evidences': {
                'semantic': evidences_sem,
                'syntactic': evidences_syn
            }
        }
