#TF-IDF based summary clustering

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KDTree

# Reading the dataset - we process our data based on summaries
df = pd.read_csv('../data/summarized.csv')

# Counting the frequency of occurrence of each word
count_vector = CountVectorizer()
train_counts = count_vector.fit_transform(df.summarized)
# Print something useful
print(train_counts.shape)
print(count_vector.vocabulary_.get("Manager"))

# Using tf-idf to reduce the weight of common words
tfidf_transform = TfidfTransformer()
train_tfidf = tfidf_transform.fit_transform(train_counts)
tree_data = np.array(train_tfidf.toarray())
kdtree = KDTree(tree_data, leaf_size=5)
df['tfidf'] = list(train_tfidf.toarray())

# Input some text
search_text = input("Enter search string:- ")
# Get the tf-idf equivalents to search
query = df['tfidf'][df['summarized'].str.contains(search_text,case=False,regex=True)].tolist()
if not query:
    search_text = "customer"
    print("No text, defaulting to ",search_text)
    query = df['tfidf'][df['summarized'].str.contains(search_text,case=False,regex=True)].tolist()

# Using KDTree to get K similar documents
K = 5
distance, idx = kdtree.query(query, k=K)
for i, value in list(enumerate(idx[0])):
    print("Job Tile : {}".format(df['title'][value]))
    print("Distance : {}".format(distance[0][i]/K))
    print("Job Summary : {}".format(df['summarized'][value]))