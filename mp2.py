import json
from nltk.corpus import stopwords
import re
import os
import string
import time
from multiprocessing.dummy import Pool as ThreadPool 
import pandas as pd


def pre_proc_text(post):

    def extract_hashtags(caption):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tags.append(m.group(0))
            return ""

        caption = regexp.sub(repl, caption.lower())
        return caption, tags

    def extract_mentions(caption):
        regexp = re.compile(r"(?:@)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tags.append(m.group(0))
            return ""

        caption = regexp.sub(repl, caption.lower())
        return caption, tags

    def remove_punct(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(text):
        text = re.split('\W+', text)
        return text
    
    def remove_stopwords(text, stopword):
        text = [word for word in text if word not in stopword]
        return text
    

    # caption = re.split('\W+', post[1])
    caption, tags = extract_hashtags(post[1])
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopwords)
    caption = ' '.join(caption)
    caption = caption.rstrip()
    
    return [post[0], post[1], caption]


start = time.time()
pages = []
list2 = []



path = "/root/dev/scrapeimport"
stopwords = stopwords.words('english')
dirs = []
for dir in os.scandir(path):
    for file in os.scandir(dir):
        with open(file, 'r') as fp:
            page = json.loads(fp.read())
            for edge in page['edges']:
                try:
                    id = edge['node']['id']
                except:
                    id = ""
                try:
                    caption = edge['node']['edge_media_to_caption']['edges'][0]['node']['text']
                except:
                    caption = ""
                list2.append([id, caption])
    

print(len(list2))
print(time.time() - start)

start = time.time()

from multiprocessing import Process, Manager, Pool

captions = []
pool = Pool(3)
captions = [x for x in pool.map(pre_proc_text, list2) if x is not None]
print(len(captions))
print(time.time() - start)
start = time.time()
output_file = "output2.csv"
df1 = pd.DataFrame(captions, columns=["id", "og_caption", "mod_caption"])
df2 = pd.read_csv(
    "/root/dev/projects/scrape/data/mlbiased.csv", header=None)
df2.columns = ["mod_caption", "target"]
df1.id = df1.id.str.strip()
df1.og_caption = df1.og_caption.str.strip()
df1.mod_caption = df1.mod_caption.str.strip()
df1.mod_caption = df1.mod_caption.astype(str)
# df1.mod_caption = df1.mod_caption.str.encode("utf-8")

df2.mod_caption = df2.mod_caption.astype(str)
df2.mod_caption = df2.mod_caption.str.strip()
df2.target = df2.target.astype(str)


df3 = pd.merge(df1, df2, how="left", on="mod_caption") 

df3.to_csv(output_file, header=True)

print(time.time() - start)
