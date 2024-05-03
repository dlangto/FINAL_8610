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

data = dataset["reviewtext"]
target = dataset["stars"]
lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = text.translate(str.maketrans("","", string.punctuation))
	#remove things we don't want in the analysis- this is punctuation
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		#make it all lowercase
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			#groups words together by making the word back to its most basic form
			result.append(word_processed)
	return result

count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)
#turns it into something we can analyze
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

#Put the reviews into sample_new.csv. Run the training file to train the data. Run the prediction file to get the prediction. The prediction and prediction probabilities will be found in rew_reviews_with_prediction.csv.
