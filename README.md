# 🎬 Movie Sentiment Analyzer (ANN + Transformer Pipeline)

An end-to-end NLP application that analyzes movie reviews and generates a final verdict by combining traditional machine learning and modern transformer-based approaches.

---

## 🚀 Features

* 🔍 Search movies dynamically using TMDB API
* 📝 Fetch real user reviews
* 🧠 Perform sentiment analysis on:

  * Movie reviews
  * Movie overview
* 📊 Generate a final verdict using:

  * ⭐ TMDB rating
  * 💬 Review sentiment
  * 📖 Overview sentiment

---

## 🧠 Models Used

### 🤖 Transformer Pipeline (Primary Model)

* Hugging Face `pipeline("sentiment-analysis")`
* Pretrained model (no training required)
* Captures contextual meaning effectively

✔ Used in the Flask application for real-time predictions

---

### 🧠 Traditional NLP + ANN

* Text preprocessing using NLTK (tokenization, lemmatization)
* TF-IDF vectorization
* Feedforward Neural Network (TensorFlow)

✔ Included for comparison and understanding classical NLP approaches

---

## 📊 Final Verdict Calculation

The system combines three signals:

* Review sentiment score
* Overview sentiment score
* TMDB rating (normalized to 0–1)

### Formula:

```
Final Score = mean(review_score, overview_score, rating / 10)
```

---

### 🎯 Verdict Logic

| Score Range | Verdict            |
| ----------- | ------------------ |
| ≥ 0.7       | Worth Watching     |
| 0.4 – 0.7   | Watchable          |
| < 0.4       | Not Worth Watching |

---

## 🖥️ Tech Stack

* Python
* Flask
* Hugging Face Transformers
* TensorFlow / Keras
* Scikit-learn
* NLTK
* TMDB API

---

## 📂 Project Structure
Movie-Sentiment-Analyzer/
├── app.py
├──models/
│ ├── ann.py
│ ├── ann_training.py
│ ├── ann_model.py
│ ├── pipeline.py
│ ├── tfidf.pkl
│ └── requirements.txt
├── templates/
│ └── index.html
|──dataset
│ └──imdb.txt   
└── README.md

## 🔄 Model Switching

By default, the Flask app uses the **Transformer pipeline**.

### ▶ Switch to ANN Model

#### Step 1: Import ANN model

```python
from model.ann import ann_model
```

#### Step 2: Replace pipeline calls

```python
review_score = pipeline.predict(reviews)
overview_score = pipeline.predict(Overview)
```

with:

```python
review_score = ann_model(reviews)
overview_score = ann_model(Overview)
```

---

## 💡 Key Insights

* Traditional models rely on manual feature extraction (TF-IDF)
* ANN captures nonlinear relationships in text data
* Transformers understand semantic context using attention mechanisms

👉 This project demonstrates the evolution of NLP approaches from classical methods to modern deep learning techniques.

---

## 📸 Example Output

### 🎬 Interstellar

* Rating: 8.4
* Review Score: 8.53
* Verdict: **Worth Watching**

---

### 🎬 Annaatthe

* Rating: 4.9
* Review Score: 0
* Verdict: **Watchable**

---

### 🎬 Disaster Movie

* Rating: 3.2
* Review Score: 0.005
* Verdict: **Not Worth Watching**

---
### Dataset
* 📥 Download the dataset [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

