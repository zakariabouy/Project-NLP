# Project-NLP: Spam Detection – Summary Report

Project: SMS Spam Detection — INSEA Projet Statistiques Multivariées 2025
Dataset: data/X_train.npy, data/X_test.npy (texts), data/Y_train.npy, data/Y_test.npy (labels); BERT embeddings precomputed as data/embeddings_train.npy, data/embeddings_test.npy.
Goal: Compare text representations (Count, TF-IDF, Word2Vec, BERT), evaluate classifiers, visualize structure, assess data drift; include unsupervised clustering.

Methods
- Features: CountVectorizer; TfidfVectorizer(max_features=3000); Word2Vec (GoogleNews-300 average per SMS); BERT (bert-base-uncased; sentence embedding = mean of last hidden state).
- Models: MultinomialNB; Logistic Regression; LDA/QDA; SVM (linear).
- Visualization: PCA (2D), t-SNE (2D).
- Drift: PCA reduction + per-component t-tests; adversarial classifier (train vs test) ROC-AUC.
- Unsupervised: GMM on PCA(10) of BERT embeddings; ARI, log-likelihood, BIC; likelihood-ratio test.

Key Results
- Count + MultinomialNB: Accuracy ˜ 98.10%; custom NumPy NB matches scikit-learn predictions exactly.
- TF-IDF + SVM (linear): Accuracy ˜ 98.13%; F1 ˜ 0.926; best TF-IDF baseline.
- Word2Vec averaging: Underperforms for SMS due to slang/OOV and loss of order; TF-IDF preferred.
- BERT + Logistic Regression: Spam precision ˜ 0.9917; F1 ˜ 0.9741; strongest overall.
- Visualization: PCA on TF-IDF captures little variance; t-SNE reveals clearer clusters; BERT improves separability.

Data Drift
- T-tests (PCA components): Low p-values indicate mean shifts when present.
- Adversarial AUC: High AUC ? drift; ~0.5 ? stable distributions. Use as operational monitor.

Unsupervised
- GMM comparison: Full covariance favored by LRT and BIC; highest ARI with full.
- Interpretation: BERT embedding dimensions correlated within clusters; full covariance is necessary.

Conclusions
- Best supervised: BERT embeddings + Logistic Regression.
- Baseline: TF-IDF + Linear SVM remains competitive and simple.
- Ops: Monitor drift (AUC, PCA+t-tests); retrain on drift.

Submission Notes
- Notebook name: groupe_X.ipynb only.
- Run all cells: ensure clean execution and no errors.
- Delivery: Dropbox link in notebook; latest upload graded.
