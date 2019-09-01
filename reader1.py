from contextlib import contextmanager

from instaloader import Instaloader
from instaloader import InstaloaderContext, Post
from instaloader.exceptions import *
from instaloader.structures import (Highlight, JsonExportable, Post, PostLocation, Profile, Story, StoryItem,
                         save_structure_to_file, load_structure_from_file, PostComment, PostCommentAnswer)
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Tuple, Iterator, List, Optional, Union

import random 
import requests
import requests.utils
import json
import time
from DB import Database
import re

def load_from_files(path, start_tag):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            resp_json = json.loads(f)
            yield from (Post(L.context, edge['node']) for edge in resp_json['edges'])
        
proxylist = []
with open("goodproxies.json","r") as file:   
    proxylist = json.load(file)


L = Instaloader()


if len(sys.argv) >= 1:
    path=sys.argv[1]
    start_tag = sys.argv[2]
    
if os.path.isdir(path):
    print(str(path))
    for post in load_from_files(path,start_tag):
        values = (post.mediaid, post.date_utc)
