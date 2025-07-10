# 🎬 IMDB Movie Review Sentiment Analysis using Word Embeddings

## 📌 Problem Statement

In the fast-evolving entertainment industry, understanding audience sentiments is critical for refining content and optimizing marketing strategies. Manually analyzing thousands of reviews is not scalable or reliable. This project develops a sentiment analysis model using advanced word embedding techniques—Word2Vec and GloVe—to classify IMDB movie reviews into positive or negative categories.

---

## 📊 Dataset

- **File**: `2.2+imdb_10K_sentimnets_reviews.csv`
- **Columns**:
  - `review`: Textual content of the movie review
  - `sentiment`: Binary label (0 for negative, 1 for positive)

---

## 🛠️ Tools & Libraries

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- nltk, gensim, sklearn
- Word2Vec, GloVe, Logistic Regression, XGBoost, FastText

---

## 🔁 Project Workflow

```python
# 1. Install Required Libraries
!pip install unidecode==1.4.0 gensim==4.3.3 zeugma==0.41 fasttext==0.9.3 \
pandas==2.2.2 numpy==1.26.4 matplotlib==3.10.0 seaborn==0.13.2 \
nltk==3.9.1 scikit-learn==1.6.1 -q

# 2. Import Libraries
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from gensim.models import Word2Vec

# 3. Load Data
df = pd.read_csv("2.2+imdb_10K_sentimnets_reviews.csv")
df['review'] = df['review'].astype(str)

# 4. Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

df['tokens'] = df['review'].apply(clean_text)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['sentiment'], test_size=0.2, random_state=42)

# 6. Word2Vec Embedding
w2v_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=2, workers=4)

def get_avg_vector(tokens, model, k=100):
    valid_tokens = [t for t in tokens if t in model.wv]
    if not valid_tokens:
        return np.zeros(k)
    return np.mean(model.wv[valid_tokens], axis=0)

X_train_vec = np.array([get_avg_vector(tokens, w2v_model) for tokens in X_train])
X_test_vec = np.array([get_avg_vector(tokens, w2v_model) for tokens in X_test])

# 7. Model Training and Evaluation
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
y_pred = lr.predict(X_test_vec)

print(classification_report(y_test, y_pred))
```

---

## ✅ Results

- Achieved **~84% accuracy** using Word2Vec embeddings with Logistic Regression.
- Word embeddings significantly improved classification performance over basic bag-of-words.
- Models can generalize well on unseen reviews.

---

## 📁 Repository Structure

```
├── Hands_on_Word2Vec_GloVe_Notebook.ipynb
├── 2.2+imdb_10K_sentimnets_reviews.csv
├── README.md
```

---

## 📌 Key Insights

- Pre-trained embeddings like Word2Vec and GloVe enhance model performance on text data.
- Cleaning and normalizing text (tokenization, lemmatization, stopword removal) is essential.
- Averaging word vectors is a simple yet effective way to represent document-level semantics.

---

## 👨‍💻 Author

**Suhaib Khalid**  
AI & ML Enthusiast 

---

## 📝 License

This project is licensed under the MIT License.

