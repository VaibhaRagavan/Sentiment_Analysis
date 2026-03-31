import pandas as pd,re,nltk,pickle 
##getting data and label
data=pd.read_csv("IMDB Dataset.csv")
X=data['review']
Y=data['sentiment']
Y=Y.map({'positive':1,'negative':0})

##preproccsing the data
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer=WordNetLemmatizer()
stop_word=set(stopwords.words("english"))
stop_word.discard("not")

##creating textpreprocess as a function so we can use for new datas
def text_processing(text):
    if isinstance(text,str):
        t=[text]
    else:
        t=text
    corpus=[]
    for i in t:
        sent=re.sub('[^a-zA-Z]',' ',i);
        sent=sent.lower();
        word=nltk.word_tokenize(sent);
        lemmatized_word=[];
        for words in word:
            if words not in stop_word:
                lemmatized_word.append(lemmatizer.lemmatize(words))
        cleaned_sentence=' '.join(lemmatized_word)
        corpus.append(cleaned_sentence) 

    if isinstance (text,str):
        return corpus[0]
    else:
        return corpus
   
clean_data=text_processing(X)


## converting the cleaned data into vector 

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=5000)
data_vec=cv.fit_transform(clean_data)
pickle.dump(cv, open("models/tfidf.pkl", "wb"))

##test and train the data 
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(data_vec,Y,test_size=0.3,random_state=45,stratify=Y)


### artifical nueron network
import tensorflow as tf 
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
model=Sequential()
model.add(Dense(units=70, activation='relu', input_shape=(X_train.shape[1],)))#input layer
model.add(Dropout(0.3))
model.add(Dense(units=50,activation='relu'))#hidden layer 1
model.add(Dense(units=25,activation='relu'))#hidden layer2
model.add(Dense(units=1,activation='sigmoid'))#output layer

##compile the neurons
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train.toarray(),Y_train,epochs=15,batch_size=20)
model.save("models/ann_model.h5")
print("✅ Model and vectorizer saved!")
