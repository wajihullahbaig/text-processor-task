# Tag entities at their locations
import pandas as pd
import spacy

nlp = spacy.load("xx_ent_wiki_sm") # multi-language corpus
nlp.add_pipe(nlp.create_pipe('sentencizer'))
df = pd.read_csv("../data/jobtitles.csv", names=["title", "description"], nrows=10)


def get_str_index(string):
    try:
        doc = nlp(string.rstrip())
        for ent in doc.ents:
            print(ent.text, ent.start_char - ent.sent.start_char, ent.end_char - ent.sent.start_char, ent.label_)

    except Exception as err:
        print(f"Error for {string} : {err}")


df.apply(lambda row: get_str_index(row['description']), axis=1)
