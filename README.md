# text_processor_task
A text processing task for NER using Spacy, Summarization using HF, searching summarization using TF-IDF

# Task 1
NER using Spacy
1- Create the dataset for training
2- In the dataset create entities {title} that are auto marked at the positions in the description text. 
3- Train and evaluate on train/test datasets

Shortcomings
I see from the approach is the multi-lingual approach as the training data is a mix of languages, English and presumably German.
The model can be improved if some preprocessing of job titles is done.
The test set does not work as well as I would have hoped to, I am not sure if this is because of the language or the test set is completely different.
Maybe I need to train the model a bit more. But I am short on time to actually explore and dig down.
References: 
https://soumilshah1995.blogspot.com/2021/04/entity-recognition-extract-information.html 
https://manivannan-ai.medium.com/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6

# Task 2
Hugginface summarization pipelines. Some basic preprocessing and saving the summaries back to the output file. Some minor pre/post processing to speed up the whole process. I observed this though I was using a GPU but for some reason it was extremely slow. 

# Task 3
So what should we do with the summarized data? How about making it searchable? For that I used TF-IDF features of the summarized text and populated a KD-Tree with a bag of words (TF-IDF) vectors. One single call for a search from a user input returns text that seems reasonable. 
Here the language is not constrained as TF-IDF does not care. :)
