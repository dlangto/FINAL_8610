import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv('dataset.csv')

# Randomly select 20% of the dataset
subset_size = int(0.2 * len(data))
subset_indices = np.random.choice(data.index, subset_size, replace=False)
subset_data = data.loc[subset_indices]
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
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
subset_data['positive'] = subset_data['reviewtext'].apply(lambda x: simple_call("In a scale of 1 to 3, how positive is the following review for a programmer: \"" + x + "\", answer in one number."   ))


# Load your dataset (replace 'programmer_dataset.csv' with your dataset file)
data = pd.read_csv('sample_new.csv')


print("Subset of Data with OpenAI Scores:")
print(subset_data)
subset_data.to_csv('dataset_processed.csv')

#Import the dataset with the stars and reviews. The machine will select 20% of the reviews and make a prediction for each of those selected. 