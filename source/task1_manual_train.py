import random
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split

spacy.prefer_gpu()


# we perform a manual training on custom dataset

class TrainDataGenerator(object):
    def __init__(self, text):
        self.text = text
        self.entities = []

    # Create the entity positions and add them
    def create_and_add_entity(self, searchTerm='', entity_name=''):
        start_index = self.text.find(searchTerm)
        if start_index != -1:
            data_entity = (start_index, start_index+len(searchTerm)-1, entity_name)
            self.entities.append(data_entity)
            return True
        else:
            return False

    def get_unique_entity(self):

        entity_tem = {"entities": self.entities}
        data = (self.text, entity_tem)

        entities = entity_tem.get("entities")

        # check if entity first index is overlapping with another one
        for i in range(0, len(entities)):
            for j in range(i + 1, len(entities)):

                StartIndex1 = entities[i][0]
                endIndex1 = entities[i][1]

                StartIndex2 = entities[j][0]
                endIndex2 = entities[j][1]

                if StartIndex1 in range(StartIndex2, endIndex2):
                    return None
                if endIndex2 in range(StartIndex2, endIndex2):
                    return None

        return data


def remove_special_chars(string):
    # For now, we are just removing digits
    string = ''.join([i for i in string if not i.isdigit()])
    return string


def prepare_train_date(df_train):
    train_data = []
    for index, row in df_train.iterrows():
        title = remove_special_chars(row.title)
        #text = title + " " + row.description  # No need to add title but this is only to check if our model can acutally find something.
        title = row.title
        text = row.description
        _helper = TrainDataGenerator(text=text)
        if _helper.create_and_add_entity(entity_name='title', searchTerm=title):
            response = _helper.get_unique_entity()
            if response:
                train_data.append(response)
    return train_data


class Model(object):

    def __init__(self, modelName="testmodel"):
        self.nlp = spacy.blank("en") # Create a blank piple line for english
        self.modelName = modelName

    def train(self, training_data, output_dir=None, n_iters=80):

        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(ner, last=True)

        # otherwise, get it so we can add labels
        else:
            ner = self.nlp.get_pipe("ner")

        # add labels
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training - Why?
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe
                       for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        with  self.nlp.disable_pipes(*other_pipes):  # only train NER
            # reset and initialize the weights randomly – but only if we're
            # training a new model
            self.nlp.begin_training()
            for itn in range(n_iters):
                print("Iteration : {} ".format(itn))
                random.shuffle(training_data)
                losses = {}

                # batch up the examples using spaCy's minibatch

                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))

                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorize data
                        losses=losses,
                    )

                print("Losses", losses)
                if losses.get("ner") < 300.0:  # save only models that actually do something
                    self.nlp.to_disk(self.modelName)
                    l = losses.get("ner")
                    print(f'Saving model at a loss of {l} ')


def train(df_train):
    training_data = prepare_train_date(df_train)
    model = Model(modelName='../model/jobposting')
    model.train(training_data=training_data, n_iters=200)


def test(df_test):
    try:
        nlp = spacy.load("../model/jobposting")
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        for index, row in df_test.iterrows():
            doc = nlp(row["description"])
            for ent in doc.ents:
                print("<Label>:", ent.label_, "\n<Text>:", ent.text, "\n<Start pos>:",
                      ent.start_char - ent.sent.start_char, " <End pos>:",
                      ent.end_char - ent.sent.start_char)

    except Exception as err:
        print(f"Error! {err}")


if __name__ == "__main__":
    df = pd.read_csv("../data/jobtitles.csv", names=["title", "description"], nrows=1500)
    train_df, test_df = train_test_split(df, test_size=0.3)
    #train(train_df)
    test(test_df)
