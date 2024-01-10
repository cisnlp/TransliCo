import os
from uroman import Uroman
import pandas as pd


def create_transliteration_data(file_path):

    directory, filename = os.path.split(file_path)
    print(filename)

    assert os.path.isfile(file_path), f"Input file path {file_path} not found"
    if (os.path.exists(f"{directory}/text_transliterations.csv") and "without" in filename) or \
    (os.path.exists(f"{directory}/text_transliterations_with_latn.csv") and "without" not in filename):
        pass
    else:
        roman = Uroman()
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = f.read().splitlines()
        transliterations = roman.romanize(sentences, './temp')
        assert len(sentences) == len(transliterations)
        num_sentences = len(sentences)
        # write transliterations:
        examples = []
        data_df = pd.DataFrame({'text': sentences, 'transliteration': transliterations})

        if "without" in filename:
            data_df.to_csv(f"{directory}/text_transliterations.csv", index=False)
            print(f"Saved in {directory}/text_transliterations.csv")
        else:
            data_df.to_csv(f"{directory}/text_transliterations_with_latn.csv", index=False)
            print(f"Saved in {directory}/text_transliterations_with_latn.csv")

create_transliteration_data("/mounts/data/proj/ayyoobbig/transliteration_modeling/train_data/1000LM.txt")
