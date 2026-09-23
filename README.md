# Sentiment Analysis on French Movie Reviews (AlloCiné)

Binary sentiment classification (positive / negative) of French movie reviews from the
[AlloCiné dataset](https://huggingface.co/datasets/tblard/allocine), comparing classical
machine-learning baselines with recurrent neural networks, including a BiLSTM with a
custom Bahdanau attention layer.

**Authors:** Ryhem Mekni, Jafra Chairi

## Results (test set, 20,000 reviews)

| Model                        | Accuracy   | Precision  | Recall     | F1         |
|------------------------------|------------|------------|------------|------------|
| TF-IDF + Logistic Regression | 0.9406     | 0.9354     | 0.9412     | 0.9383     |
| **TF-IDF + LinearSVC**       | **0.9430** | **0.9368** | **0.9450** | **0.9408** |
| BiLSTM (learned embeddings)  | 0.9353     | 0.9343     | 0.9305     | 0.9324     |
| BiLSTM + Bahdanau Attention  | 0.9375     | 0.9289     | 0.9417     | 0.9353     |

**Key takeaways**

- The TF-IDF + LinearSVC baseline performs best, and it is also the fastest model to train.
- Adding attention improves the plain BiLSTM slightly (+0.2 pt accuracy, +1.1 pt recall).
- Both neural models overfit quickly. Their best validation loss comes at epoch 2, and early stopping ends training at epoch 5.

## Dataset

| Split      | Reviews |
|------------|---------|
| Train      | 160,000 |
| Validation | 20,000  |
| Test       | 20,000  |

Labels: `0` = negative, `1` = positive. The two classes are roughly balanced.

## Approach

### Preprocessing
- Lowercasing and whitespace normalization (`clean_text`).
- **Classical models:** `TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, strip_accents="unicode")`.
- **Neural models:** Keras `Tokenizer` with a 50,000-word vocabulary and an `<OOV>` token. Sequences are padded or truncated to 200 tokens.

### Models
1. **TF-IDF + Logistic Regression:** `solver="saga"`, `max_iter=1000`.
2. **TF-IDF + LinearSVC:** `max_iter=5000`.
3. **BiLSTM:**
   `Embedding(50000, 100) → BiLSTM(128) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid)`, about 5.25M parameters.
4. **BiLSTM + Bahdanau Attention:** the BiLSTM returns the full sequence. A `GlobalAveragePooling1D` output serves as the query, and a custom `BahdanauAttention(64)` layer produces a context vector over the 200 time steps. The classification head is the same as model 3. About 5.28M parameters.

The neural models are trained with Adam (lr = 1e-3), binary cross-entropy, batch size 256, up to 8 epochs, with `EarlyStopping(patience=3, restore_best_weights=True)` on `val_loss`.

## Repository structure

```
.
├── sentiment_analysis_classification_.ipynb   # Full code with executed outputs
├── report.pdf                                # Project report
└── README.md
```

## Getting started

```bash
pip install datasets pandas numpy scikit-learn tensorflow matplotlib seaborn
jupyter notebook
```

Open the notebook and run the cells in order. The dataset downloads automatically from
Hugging Face the first time you run it.

> On CPU, each BiLSTM epoch takes about 7–13 minutes on the full training set. A GPU is recommended.

## Future work

- Visualize the attention weights on real reviews to interpret what the model focuses on.
- Use pre-trained French embeddings (e.g. fastText) or French Transformers (CamemBERT, FlauBERT).
- Add significance testing (e.g. McNemar's test), since the gaps between models are small.
- Tune regularization to reduce overfitting in the neural models.
