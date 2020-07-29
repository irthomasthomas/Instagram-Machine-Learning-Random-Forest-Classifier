import json
from nltk.corpus import stopwords
import re
import os
import string
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


list2 = []


# scans each folder in path for instagram files 
# and adds each 
#  id and caption to list2
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

## process list2 caption text and add to captions list
from multiprocessing import Process, Manager, Pool
captions = []
pool = Pool(3)
captions = [x for x in pool.map(pre_proc_text, list2) if x is not None]
print(len(captions))
##

### Make a df of csv file of og and mod captions
df1 = pd.DataFrame(captions, columns=["id", "og_caption", "mod_caption"])
df1.id = df1.id.str.strip()
df1.og_caption = df1.og_caption.str.strip()
df1.mod_caption = df1.mod_caption.str.strip()
df1.mod_caption = df1.mod_caption.astype(str)
# df1.mod_caption = df1.mod_caption.str.encode("utf-8")

# make 2nd df of mlbiased data
df2 = pd.read_csv(
    "/root/dev/projects/scrape/data/mlbiased.csv", header=None)  # 20k cleaned and tagged / 3k 1s
df2.columns = ["mod_caption", "target"]
df2.mod_caption = df2.mod_caption.astype(str)
df2.mod_caption = df2.mod_caption.str.strip()
df2.target = df2.target.astype(str)
# TODO: df2.ods
# merge df 1 & 2
output_file = "output2.csv"  # 55k rows/4k tagged 1s id| og_caption| mod_caption| target 
df3 = pd.merge(df1, df2, how="left", on="mod_caption") 
# save df3 to csv
df3.to_csv(output_file, header=True)

