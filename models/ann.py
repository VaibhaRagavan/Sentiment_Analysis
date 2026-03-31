import pickle
import re
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load model + vectorizer
model = load_model("models/ann_model.h5")
cv = pickle.load(open("models/tfidf.pkl", "rb"))
# Preprocessing
def text_processing(text):
    if isinstance(text,str):
        t=[text]
    else:
        t=text
    corpus=[]
    lemmitizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    for i in t:
        sent=re.sub('[^a-zA-Z]',' ',i);
        sent=sent.lower();
        word=nltk.word_tokenize(sent);
        lemmitized_word=[];
        for words in word:
            if words not in stop_words:
                lemmitized_word.append(lemmitizer.lemmatize(words))
        cleaned_sentence=' '.join(lemmitized_word)
        corpus.append(cleaned_sentence) 

    if isinstance (text,str):
        return corpus[0]
    else:
        return corpus


##testing with new data
def ann_model(test_data):
    pred_score=[]
    for items in test_data:
        revised_data=text_processing(items)
        test_data_vec=cv.transform([revised_data])
        pred=model.predict(test_data_vec.toarray())
        pred_value=pred[0][0]
        pred_score.append(pred_value)

    return np.mean(pred_score) if pred_score else 0
