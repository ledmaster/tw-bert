#!/usr/bin/env python3

import torch
import re
from collections import Counter
import tw_bert_v2

def original_score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len, k1=1.2, k3=8., b=0.75):
    """Original non-vectorized implementation for comparison"""
    query = re.findall(r"[a-z0-9']+", query)
    
    if isinstance(query_tf_vec, list):
        query_tf_vec = torch.tensor(query_tf_vec, device=term_weights.device, dtype=torch.float32)
    else:
        query_tf_vec = query_tf_vec.to(term_weights.device)
    
    weighted_query_terms = term_weights * query_tf_vec
    num_docs = len(corpus)
    
    query_idf = {}
    for term in query:
        document_frequency = sum([1 for doc_tf in corpus if term in doc_tf])
        query_idf[term] = torch.log(torch.tensor((num_docs - document_frequency + 0.5)/(document_frequency+0.5) + 1))
    
    doc_scores = list()
    for doc_tf in corpus:
        doc_len = sum(doc_tf.values())
        doc_tf_vec = torch.tensor([doc_tf.get(term, 0) for term in query], device=term_weights.device, dtype=torch.float32)
        numerator = doc_tf_vec * (k3 + 1) * weighted_query_terms
        normalization_factor = k1 * ((1-b) + b * doc_len/avg_doc_len) + doc_tf_vec
        denominator = (k3 + weighted_query_terms) * normalization_factor
        idf = torch.tensor([query_idf[term] for term in query], device=term_weights.device, dtype=torch.float32)
        doc_scores.append(torch.sum(idf * numerator/denominator))
    
    return torch.stack(doc_scores)

def test_vectorization():
    torch.set_default_device('cpu')  # Use CPU for testing
    
    # Test data
    query = "machine learning algorithms"
    query_tf_vec = [1, 1, 1]  # term frequencies for each query term
    
    # Mock corpus with term frequencies
    corpus = [
        {"machine": 2, "learning": 1, "algorithms": 1, "data": 3},
        {"machine": 1, "learning": 2, "neural": 1, "networks": 1},
        {"algorithms": 3, "sorting": 2, "computer": 1},
        {"data": 2, "science": 1, "machine": 1}
    ]
    
    # Mock term weights (simulating BERT output)
    term_weights = torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32)
    avg_doc_len = 4.0
    
    # Test both implementations
    print("Testing vectorized score_vec...")
    
    try:
        vectorized_scores = tw_bert_v2.score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len)
        original_scores = original_score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len)
        
        print(f"Vectorized scores: {vectorized_scores}")
        print(f"Original scores:   {original_scores}")
        
        # Check if they're close (allowing for floating point differences)
        if torch.allclose(vectorized_scores, original_scores, rtol=1e-5, atol=1e-6):
            print("✅ PASS: Vectorized implementation produces identical results!")
            return True
        else:
            print("❌ FAIL: Results differ!")
            print(f"Max difference: {torch.max(torch.abs(vectorized_scores - original_scores))}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_vectorization()