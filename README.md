![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Fake News Detection — NLP Classification Project

## About

This project builds a machine learning classifier that distinguishes **fake** from **real** news articles based on their title and text content. It uses classic NLP techniques (TF-IDF, Bag of Words, Word2Vec embeddings) combined with several classification algorithms (Logistic Regression, Naive Bayes, Random Forest), and compares their performance on a held-out test set.

A key part of this project was a thorough **data leakage investigation**: several columns in the raw dataset (`subject`, `date`, and source tags inside `text`) turned out to perfectly predict the label due to how the data was collected — not because of any real signal about "fakeness". These were identified and excluded before modeling, so the final classifier learns from genuine content patterns rather than shortcuts.

### Sample Predictions

label                                              title  \
0      1  UK's May 'receiving regular updates' on London...   
1      1  UK transport police leading investigation of L...   
2      1  Pacific nations crack down on North Korean shi...   
3      1  Three suspected al Qaeda militants killed in Y...   
4      1  Chinese academics prod Beijing to consider Nor...   

                                                text    subject  \
0  LONDON (Reuters) - British Prime Minister Ther...  worldnews   
1  LONDON (Reuters) - British counter-terrorism p...  worldnews   
2  WELLINGTON (Reuters) - South Pacific island na...  worldnews   
3  ADEN, Yemen (Reuters) - Three suspected al Qae...  worldnews   
4  BEIJING (Reuters) - Chinese academics are publ...  worldnews   

                  date  
0  September 15, 2017   
1  September 15, 2017   
2  September 15, 2017   
3  September 15, 2017   
4  September 15, 2017   
(4956, 5)


This project was not deployed as a live app or API — it runs locally as a Jupyter Notebook.

## Problem Statement

- **Dataset used:** `data.csv` (~40,000 labeled news articles) and `validation_data.csv` (~5,000 unlabeled articles)
- **Task:** Binary text classification — predict whether a news article is fake (0) or real (1)
- **Goal:** Build a classifier trained and evaluated on `data.csv`, then apply it to generate predictions for `validation_data.csv`

## Dataset

- **Source:** Provided via the [Ironhack NLP Challenge repository](https://github.com/ironhack-labs/project-nlp-challenge) (based on the widely-used "Fake and Real News" dataset)
- **Size:** 39,942 labeled articles (`data.csv`) + 4,956 unlabeled articles for final prediction (`validation_data.csv`)
- **Classes:** Balanced — 50.07% real, 49.93% fake
- **Columns:** `label` (0=fake, 1=real), `title`, `text`, `subject`, `date`

## Model Architecture

**Pipeline overview:**

1. **EDA & Data Leakage Analysis** — identified that `subject`, `date`, and a "(Reuters) -" source tag in `text` leak the label almost perfectly (data collection artifacts). These were excluded from modeling; the Reuters prefix was stripped from `text`.
2. **Train/Test Split** — 80/20, stratified by label
3. **Preprocessing** — tokenization, lowercasing, punctuation & stopword removal, lemmatization (with POS tagging)
4. **Feature Extraction** — three approaches compared: TF-IDF, Bag of Words, and Word2Vec (averaged word embeddings)
5. **Classification** — Logistic Regression, Naive Bayes, and Random Forest compared across feature sets

```
Raw text ──▶ Clean (remove source tags) ──▶ Tokenize/Lowercase ──▶ Remove stopwords/punctuation
          ──▶ Lemmatize ──▶ Feature extraction (TF-IDF / BoW / Word2Vec) ──▶ Classifier ──▶ Prediction
```

## Results

Five model combinations were trained and evaluated on the held-out test set:

| Model | Feature Extraction | Algorithm | Accuracy |
|---|---|---|---|
| A | TF-IDF | Logistic Regression | 97.58% |
| B | TF-IDF | Naive Bayes | 91.61% |
| **C** | **BoW** | **Logistic Regression** | **98.28%** ⭐ *(chosen)* |
| D | Word2Vec | Logistic Regression | 98.14% |
| E (bonus) | BoW | Random Forest | 98.30% |

**Final model: BoW + Logistic Regression.** Although Random Forest scored marginally higher (98.30% vs 98.28% — not a meaningful difference), Logistic Regression was chosen for its faster training time and interpretability.

**Estimated accuracy on `validation_data.csv`: ~98%**, based on test set performance.

Predictions on the validation set were distributed as 69.03% fake / 30.97% real.

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone git clone https://github.com/skander10/project-nlp-challenge.git
   cd project-nlp-challenge
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter gensim
   ```
3. Download required NLTK data (run once, in the notebook):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('averaged_perceptron_tagger_eng')
   ```
4. Place `data.csv` and `validation_data.csv` inside the `dataset/` folder.
5. Run `notebook.ipynb` from top to bottom.

## Project Structure

```
project-nlp-challenge/
├── dataset/
│   ├── data.csv
│   └── validation_data.csv
├── notebook.ipynb
├── output/
│   └── predictions.csv
└── README.md
```

## Tech Stack

- Python, pandas, numpy
- NLTK (tokenization, stopwords, lemmatization, POS tagging)
- scikit-learn (TF-IDF, BoW, Logistic Regression, Naive Bayes, Random Forest, metrics)
- gensim (Word2Vec)
- matplotlib / seaborn (EDA visualization)

## Future Improvements

- Combine `title` and `text` as input (currently only `text` is used; some rows had empty `text` but informative titles)
- Try additional models (SVM, gradient boosting) or fine-tune a transformer-based model (e.g. DistilBERT)
- Use a proper 3-way train/validation/test split if testing many more model configurations
- Deploy as a simple web app/API for interactive predictions

## Author

Skander — https://github.com/skander10

1. **Python Code:** Provide well-documented Python code that conducts the analysis.
2. **Predictions:** A csv file in the same format as `validation_data.csv` but with the predicted labels (0 or 1)
3. **Accuracy estimation:** Provide the teacher with your estimation of how your model will perform.
4. **Presentation:** You will present your model in a 10-minute presentation. Your teacher will provide further instructions.
