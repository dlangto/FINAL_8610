import nltk

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import json
import pandas
import kfold_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import dill as pickle
#lets us save with pickle
review_stars=[]
review_text=[]
dataset = pandas.read_csv("dataset.csv")
print(dataset)

data = dataset["profile"]
target = dataset["stars"]
lemmatizer = WordNetLemmatizer()

def pre_processing(text):
  result = []
  result.append(data)
  return result

count_vectorize_transformer = CountVectorizer()
#turns it into something we can analyze
count_vectorize_transformer.fit(data)
data = count_vectorize_transformer.transform(data)


machine = MultinomialNB()
machine.fit(data, target)

return_values = kfold_template.run_kfold(machine, data, target, 4, False)
#splits into 4, True means continuous
print(return_values)

import dill as pickle
#lets us save with pickle

with open("text_analysis_machine.pickle", "wb") as f:
  pickle.dump(machine, f)
  pickle.dump(count_vectorize_transformer, f)
  pickle.dump(lemmatizer, f)
  pickle.dump(stopwords, f)
  pickle.dump(string, f)
  pickle.dump(pre_processing, f)
  #same process as pickle, but lets us save other stuff
import pandas

import dill as pickle


with open("text_analysis_machine.pickle", "rb") as f:
  machine = pickle.load(f)
  count_vectorize_transformer = pickle.load(f)
  lemmatizer = pickle.load(f)


  

new_reviews = pandas.read_csv("dataset.csv") 



prediction = machine.predict(data)
prediction_prob = machine.predict_proba(data)
print(prediction)
print(prediction_prob)

new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)


prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
  prediction_prob_dataframe.columns[0]: "prediction_prob_1",
  prediction_prob_dataframe.columns[1]: "prediction_prob_2",
  prediction_prob_dataframe.columns[2]: "prediction_prob_3"
  })



new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)

print(new_reviews)


new_reviews = new_reviews.rename(columns={
  new_reviews.columns[0]: "text"
  })

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_2'] = round(new_reviews['prediction_prob_2'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)


new_reviews.to_csv("new_reviews_with_prediction.csv", index=False)
#this program predicts how good a programmer will be dependig on their profile picture type. 


