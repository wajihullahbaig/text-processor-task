import pandas as pd
from transformers import pipeline


def truncate(input_list, max_length):
    tokenized = [sub.split() for sub in input_list]
    tokenized = [x if len(x) <= max_length else x[:max_length] for x in tokenized]
    joined = [" ".join(x) for x in tokenized]
    return joined


# Taking first N rows only. It's pretty slow on the CPU, so we are limiting ourselves
df = pd.read_csv("../data/jobtitles.csv", names=["title", "description"], nrows=500)
df["original_length"] = df["description"].apply(len);
# Make sure we have something to summarize
df = df[df["original_length"] > 3]
df["summarized_Length"] = 0;

to_tokenize = df["description"].to_list()
# Set max_tokens to summarize, for CPU, this is still very time-consuming. Use a smaller model?
max_tokens = 250
to_tokenize = truncate(to_tokenize, max_tokens)

# Initialize the HuggingFace summarization pipeline
summarizer = pipeline("summarization", device=1)
summarized = summarizer(to_tokenize, min_length=4, max_length=max_tokens, clean_up_tokenization_spaces=True)

# store the summary
df["summarized"] = [kv['summary_text'] for kv in summarized]

print(summarized)
print(df["summarized"])

df["summarized_length"] = df["summarized"].apply(len);

print(df["summarized_length"])
print(df["original_length"])

df.to_csv("../data/summarized.csv")

## Code I could not run because of the errors below while trying a smaller model
# ValueError: Couldn't instantiate the backend tokenizer from one of:
# (1) a `tokenizers` library serialization file,
# (2) a slow tokenizer instance to convert or
# (3) an equivalent slow tokenizer class to instantiate and convert.


# from transformers import AutoModel, AutoTokenizer
#
#
# # Define the model repo
# model_name = "sshleifer/tiny-mbart"
#
#
# # Download pytorch model
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#
# # Transform input tokens
# inputs = tokenizer("Hello world!", return_tensors="pt")
#
# # Model apply
# outputs = model(**inputs)

##  Faced a divsion error whic was fixed in the following line and file
# next_indices = (next_tokens // vocab_size).long() error in generation_utils.py
