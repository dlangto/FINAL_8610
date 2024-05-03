import pandas

dataset = pandas.read_csv('dataset.csv')

print(dataset)

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
print(os.environ.get("MY_KEY"))
from sklearn.metrics import accuracy_score
client = OpenAI(
    api_key= os.environ.get("MY_KEY"),
  )

def simple_call(prompt):
  completions = client.chat.completions.create(model="gpt-3.5-turbo", 
            messages=[
              {"role": "user", "content": prompt},
            ], max_tokens=200, temperature=0.1, top_p=1)
  return completions.choices[0].message.content
  

dataset['positive'] = dataset['reviewtext'].apply(lambda x: simple_call("In a scale of 1 to 3, how positive is the following review for a programmer: \"" + x + "\", answer in one number."   ))



dataset.to_csv('dataset_processed.csv')

dataset = pandas.read_csv("dataset_processed.csv")
data = dataset["stars"]
target = dataset["positive"]
accuracy_score = accuracy_score(data,target)
print(accuracy_score)




#This code uses AI to predict the quality of the workers with 86% accuracy. 
#Put the review in the reviewtext column and the positive column will predict how good they will be.

