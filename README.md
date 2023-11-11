# Implementation of End-to-End Query Term Weighting (TW-BERT)

This repository contains an implementation of the TW-BERT model proposed in the paper [End-to-End Query Term Weighting](https://research.google/pubs/pub52462/) by Google Research.

You can find a [detailed explanation of the code here](https://forecastegy.com/posts/tw-bert-end-to-end-query-term-weighting-pytorch/).

TW-BERT works by learning to assign weights to parts of a search query, specifically n-grams.

For example, if we had the queries:

1. "Machine learning applications in healthcare"
2. "Machine maintenance in industrial settings"

In these two queries, the term "Machine" appears in both, but would likely be assigned different weights by the TW-BERT model. 

This is due to the surrounding context of the term in each query. 

In the first query, "Machine" is part of the phrase "Machine learning", which is a specific field of study. 

In the second query, "Machine" is used in a more general sense, referring to any type of machinery used in industrial settings. 

The TW-BERT model would assign a weight to each of these terms based on its learned parameters. 

These weights can then be used directly by a retrieval system to perform a search. 

To optimize these weights, TW-BERT uses a scoring function, like BM25, which is a ranking function used by search engines to rank matching documents according to their relevance to a given search query. 

This process is done in an end-to-end fashion, meaning the model is directly trained with the downstream scoring function.

## `tw_bert_v2.py`

This script defines the TW-BERT model and the loss function used for training. 

It uses a pre-trained BERT model to obtain contextualized wordpiece embeddings, masks these embeddings using a term mask to handle n-grams, and then predicts term weights using a linear layer and a ReLU activation function.

The loss function, `TWBERTLossFT`, is a custom function that consists of two parts: the AMSE loss and the ListMLE loss. 

The AMSE loss is a variant of the mean squared error that takes into account two thresholds for the absolute difference between the predicted and true term weights. 

The ListMLE loss is a ranking loss that encourages the model to rank relevant documents higher than non-relevant ones.

The script also includes utility functions for tokenizing a query and creating a term mask for the n-grams in the query (`token_and_mask_query`), and for scoring a query against a corpus of documents using a BM25-like scoring function (`score_vec`).

## `run_msmarco_test.py`

This script uses the MSMARCO dataset to test the implementation of the model.

The script first loads the dataset, cleans and preprocesses the data, and then trains the TW-BERT model. 

The model is trained using a variant of gradient descent, with the gradients being accumulated over multiple iterations before being applied to update the model's parameters. 

## Dependencies

The scripts require the following libraries:

- PyTorch
- Transformers
- Numpy
- NLTK
- Polars
- Joblib
- Pathlib

## Suggested Improvements

My goal was to reproduce, as closely as possible, the architecture and pipeline from the original paper.

It's supposed to be a first step towards a more optimized (and deployable) implementation.

I cared much more about the `tw_bert_v2.py` file and barely optimized the data processing and pipeline in `run_msmarco_test.py`.

One clear improvement you can do is vectorize all the inputs, batch and use `torch.utils.data.DataLoader` to load the data.

The `score_vec` function can also use more vectorization.

## Differences from Original Paper

- **No Bi-gram**: This implementation does not consider bi-grams (two-word phrases) when assigning term weights. The original paper's model consider both uni-grams (single words) and bi-grams.

- **No Query Expansion**: Query expansion, a technique used to improve retrieval performance by adding synonyms or related terms to the original query, is not used in this implementation.

- **Term Uniqueness**: The original paper is unclear on whether it uses unique terms or all terms in the query for the term weighting process. This implementation uses the original query with all terms.

- **No Score Normalization**: This implementation does not implement the score normalization process described in the original paper.

- **Pretraining**: The original paper may use a pretraining phase, where the model is trained on a large, general dataset before being fine-tuned on the specific task. The BERT model used is pretrained, but not on the MSMARCO dataset with T5 query expansion.

## References

- [End-to-End Query Term Weighting](https://research.google/pubs/pub52462/) by Google Research
- [ListMLE](https://github.com/allegro/allRank/tree/master/allrank/models/losses)
- [Gradient Accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation)

PS: if you find it useful, please 🌟 the repo, thanks!