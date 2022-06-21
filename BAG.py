from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np
import pandas as pd


def extract_datax(file):
    print("sss")
    data = pd.read_parquet(file, engine='pyarrow')
    return data


def writefile(data, file):
    return 1


def Bagofword(fo):
    my_stop_words = ["%", "&", "*", "^", "~", "!", "=", ">", "<", "|", "?"]
    vectorizer = CountVectorizer(
        stop_words=my_stop_words, min_df=0.1)
    bag = vectorizer.fit_transform(fo)
    for i in range(len(vectorizer.get_feature_names())):
        print(vectorizer.get_feature_names()[i])
    return bag.toarray()


def extract_data_token_bow(file):

    result = extract_datax(file)

    code = []
    for i in range(len(result)):
        code.append(result[i][1].replace('_\\r', ' ').replace(
            '\\r', '').replace('_\\n', '').replace('\\n', ' '))

    BOW = Bagofword(code)

    if len(result) == len(BOW):
        print("same")
        data = []
        for i in tqdm(range(len(result))):
            temp = []
            temp.append(result[i][0])
            data.append(list(BOW[i]))

        print(len(data[1][1]))
        writefile(data, "Data\\BOW.txt")
