import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import json

FILE_DIR = 'HiTin/data/data.csv'
total_len = []
np.random.seed(7)

english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]

# Function to clean and tokenize text
def clean_str(string):
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def split_train_dev_test():
    f = open('custom_dataset_total.json', 'r')
    data = f.readlines()
    f.close()
    np_data = np.array(data)
    print(np_data.shape[0])
    id = [i for i in range(np_data.shape[0])]
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    f = open('custom_dataset_total_train.json', 'w')
    f.writelines(train)
    f.close()
    f = open('custom_dataset_total_test.json', 'w')
    f.writelines(test)
    f.close()
    f = open('custom_dataset_total_val.json', 'w')
    f.writelines(val)
    f.close()
    print(len(train), len(val), len(test))
    return




def create_taxonomy(df):
    taxonomy = {}
    taxonomy['Root'] = []
    for _, row in df.iterrows():
        level1 = row['Cat1']
        level2 = row['level_2']
        if level1 not in taxonomy['Root']:
            taxonomy['Root'].append(level1)
            taxonomy[level1] = {}

        if level2 not in taxonomy[level1]:
            taxonomy[level1][level2] = []

    return taxonomy

def write_taxonomy_to_file(taxonomy, filename='custom_dataset.taxonomy'):
    with open(filename, 'w') as f:
        f.write(f"Root\t" + "\t".join(taxonomy['Root']) + "\n")
        for level1 in taxonomy['Root']:
            level2_dict = taxonomy[level1]
            f.write(f"{level1}\t" + "\t".join(level2_dict.keys()) + "\n")


def df_to_dict(df):
    data = []
    for _, row in df.iterrows():
        inpt_text = str(row['Title'])+". "+str(row['Text'])
        doc = clean_str(inpt_text)
        token = [word.lower() for word in doc.split() if word not in english_stopwords and len(word) > 1]
        label = [row['Cat1'], row['level_2']]
        data.append({'token': token, 'label': label, 'topic': [], 'keyword': []})
    f = open('custom_dataset_total.json', 'w')
    for line in data:
        line = json.dumps(line)
        f.write(line + '\n')
    f.close()


print(os.getcwd())
df = pd.read_csv(FILE_DIR)
df['level_2'] = df[['Cat2','Cat3']].apply(lambda x: x.Cat2+"_"+x.Cat3,axis=1)
df_to_dict(df)
taxonomy = create_taxonomy(df)
write_taxonomy_to_file(taxonomy)
split_train_dev_test()






