import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import stanza


class GraphConstructor:
    """
    Xây dựng đồ thị ngữ nghĩa và cú pháp cho claims và evidences
    theo phương pháp ESEN
    """
    
    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Kích thước sliding window để xác định kết nối 
                        giữa các từ trong semantic graph
        """
        self.window_size = window_size
        # Khởi tạo Stanford CoreNLP qua Stanza
        # Note: Stanza không hỗ trợ MPS (Mac GPU), chỉ hỗ trợ CUDA.
        # Sẽ sử dụng CPU cho Stanza, model sẽ chạy trên MPS nếu có.
        self.nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            use_gpu=torch.cuda.is_available()  # Only use GPU if CUDA available
        )
    
    def build_semantic_graph(self, text: str) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Xây dựng semantic graph với sliding window
        
        Args:
            text: Văn bản đầu vào (claim hoặc evidence)
            
        Returns:
            adj_matrix: Ma trận kề (adjacency matrix)
            vocab: Danh sách từ vựng (unique words)
            word2idx: Ánh xạ từ -> chỉ số node
        """
        # Tokenize văn bản using the same Stanza pipeline as syntactic graph
        # so that semantic and syntactic graphs share the same tokenization
        doc = self.nlp(text)
        words = []
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text.lower())
        
        # Merge các từ lặp lại thành một node duy nhất
        unique_words = []
        word2idx = {}
        idx = 0
        
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                unique_words.append(word)
                idx += 1
        
        n_nodes = len(unique_words)
        
        # Khởi tạo ma trận kề
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Sử dụng sliding window để xác định kết nối
        for i, word in enumerate(words):
            node_i = word2idx[word]
            
            # Xét các từ trong cửa sổ
            for j in range(max(0, i - self.window_size), 
                          min(len(words), i + self.window_size + 1)):
                if i != j:
                    word_j = words[j]
                    node_j = word2idx[word_j]
                    
                    # Tạo kết nối co-occurrence
                    adj_matrix[node_i, node_j] = 1.0
                    adj_matrix[node_j, node_i] = 1.0
        
        # Thêm self-loop
        adj_matrix += np.eye(n_nodes)
        
        return torch.FloatTensor(adj_matrix), unique_words, word2idx
    
    def build_syntactic_graph(self, text: str) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Xây dựng syntactic graph sử dụng Stanford CoreNLP
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            adj_matrix: Ma trận kề có trọng số cú pháp Ã^(0)
            vocab: Danh sách từ vựng
            word2idx: Ánh xạ từ -> chỉ số node
        """
        # Parse văn bản với Stanford CoreNLP
        doc = self.nlp(text)
        
        # Trích xuất từ và dependency relations
        words = []
        dependencies = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text.lower())
            
            # Lấy syntactic dependencies
            for dep in sentence.dependencies:
                # dep = (head, deprel, dependent)
                head_idx = dep[0].id - 1  # Convert to 0-indexed
                dep_idx = dep[2].id - 1
                dep_relation = dep[1]  # Loại quan hệ cú pháp
                
                if head_idx >= 0:  # Bỏ qua ROOT
                    dependencies.append((head_idx, dep_idx, dep_relation))
        
        # Merge các từ lặp lại
        unique_words = []
        word2idx = {}
        # Map từ vị trí gốc sang node mới
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
        
        # Khởi tạo ma trận kề với trọng số cú pháp
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Tính trọng số cú pháp cho các cặp từ
        # Sử dụng dependency relations để tính trọng số
        dep_weights = defaultdict(float)
        
        for head_orig, dep_orig, relation in dependencies:
            head_new = original2new[head_orig]
            dep_new = original2new[dep_orig]
            
            # Tính trọng số dựa trên loại dependency
            # Các dependency quan trọng hơn có trọng số cao hơn
            weight = self._calculate_syntactic_weight(relation)
            
            # Cộng dồn trọng số nếu có nhiều dependency giữa cùng cặp từ
            dep_weights[(head_new, dep_new)] += weight
            dep_weights[(dep_new, head_new)] += weight
        
        # Điền trọng số vào ma trận kề
        for (i, j), weight in dep_weights.items():
            adj_matrix[i, j] = weight
        
        # Normalize trọng số (optional, tùy implementation cụ thể)
        # Thêm self-loop
        adj_matrix += np.eye(n_nodes)
        
        return torch.FloatTensor(adj_matrix), unique_words, word2idx
    
    def _calculate_syntactic_weight(self, relation: str) -> float:
        """
        Tính trọng số cú pháp dựa trên loại dependency relation
        
        Args:
            relation: Loại quan hệ cú pháp (nsubj, obj, amod, etc.)
            
        Returns:
            Trọng số từ 0.0 đến 1.0
        """
        # Các dependency quan trọng có trọng số cao
        important_deps = {
            'nsubj': 1.0,      # Nominal subject
            'obj': 1.0,        # Object
            'iobj': 0.9,       # Indirect object
            'csubj': 0.9,      # Clausal subject
            'amod': 0.8,       # Adjectival modifier
            'advmod': 0.7,     # Adverbial modifier
            'compound': 0.8,   # Compound
            'conj': 0.6,       # Conjunct
        }
        
        return important_deps.get(relation, 0.5)
    
    def process_claim_evidence(self, claim: str, evidences: List[List]) -> Dict:
        """
        Xử lý một claim và các evidence của nó
        
        Args:
            claim: Văn bản claim
            evidences: Danh sách evidences [[id, text, source], ...]
            
        Returns:
            Dictionary chứa tất cả đồ thị đã xây dựng
        """
        # Xây dựng đồ thị cho claim
        claim_sem_adj, claim_sem_vocab, claim_sem_w2i = self.build_semantic_graph(claim)
        claim_syn_adj, claim_syn_vocab, claim_syn_w2i = self.build_syntactic_graph(claim)
        
        # Xây dựng đồ thị cho từng evidence
        evidences_graphs = []
        for evi_id, evi_text, evi_source in evidences:
            evi_sem_adj, evi_sem_vocab, evi_sem_w2i = self.build_semantic_graph(evi_text)
            evi_syn_adj, evi_syn_vocab, evi_syn_w2i = self.build_syntactic_graph(evi_text)
            
            evidences_graphs.append({
                'id': evi_id,
                'source': evi_source,
                'semantic': {
                    'adj_matrix': evi_sem_adj,  # G_sem
                    'vocab': evi_sem_vocab,
                    'word2idx': evi_sem_w2i
                },
                'syntactic': {
                    'adj_matrix': evi_syn_adj,  # G_syn, tương ứng Ã^(0)_e
                    'vocab': evi_syn_vocab,
                    'word2idx': evi_syn_w2i
                }
            })
        
        return {
            'claim': {
                'text': claim,
                'semantic': {
                    'adj_matrix': claim_sem_adj,  # G^c_sem
                    'vocab': claim_sem_vocab,
                    'word2idx': claim_sem_w2i
                },
                'syntactic': {
                    'adj_matrix': claim_syn_adj,  # G^c_syn, tương ứng Ã^(0)_c
                    'vocab': claim_syn_vocab,
                    'word2idx': claim_syn_w2i
                }
            },
            'evidences': evidences_graphs
        }


# Sử dụng
if __name__ == "__main__":
    # Load dữ liệu
    with open('data/PolitiFact/json/5fold/train_0.json', 'r') as f:
        data = json.load(f)
    
    # Khởi tạo constructor
    graph_constructor = GraphConstructor(window_size=3)
    
    # Xử lý một mẫu
    sample_key = "109.json"
    sample = data[sample_key]
    
    graphs = graph_constructor.process_claim_evidence(
        claim=sample['claim_text'],
        evidences=sample['evidences']
    )
    
    # Kiểm tra kết quả
    print(f"Claim semantic graph: {graphs['claim']['semantic']['adj_matrix'].shape}")
    print(f"Claim syntactic graph: {graphs['claim']['syntactic']['adj_matrix'].shape}")
    print(f"Number of evidences: {len(graphs['evidences'])}")
    print(f"Evidence 0 semantic graph: {graphs['evidences'][0]['semantic']['adj_matrix'].shape}")
