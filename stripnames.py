from contextlib import contextmanager
import os
from datetime import datetime
import random 
import json
import time
from DB import Database
import itertools
import csv
import sqlite3

def update_end_cursor(end_cursor,hashtag,has_next_page):
    inputs = (end_cursor,hashtag,has_next_page,datetime.now())
        
    sql = """ REPLACE INTO scraper_status(end_cursor,hashtag,has_next_page,update_date)
                VALUES(?,?,?,?) """
    with Database("instagram.sqlite3") as db:
        db.execute(sql,inputs)
    
def get_end_cursor(hashtag):
    sql = """ SELECT end_cursor 
            FROM scraper_status 
            WHERE hashtag = ? 
            LIMIT 1 """
    with Database("instagram.sqlite3") as db:
        db.execute(sql,(hashtag,))
        end_cursor = db.fetchone()
    return end_cursor[0]

def get_proxy_db():
    with Database("instagram.sqlite3") as db:
        sql = ("""SELECT ip 
            from proxy limit 20 """)
        db.execute(sql)
        proxieslist = db.fetchall()
    return proxieslist

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def has_num(s):
    return any(i.isdigit() for i in s)

proxylist = []
newlist = []
# with open("goodproxies.json","r") as file:   
    # proxylist = json.load(file)

with Database('instagram.db') as db:
    sql = """ select post_id, username, date, owner_id, text from comment WHERE text glob '*[0-9]*' """
    # db.row_factory = dict_factory
    db.row_factory = sqlite3.Row
    db.execute(sql)
    # col_name_list = [tuple[0] for tuple in db.cursor.description] # fetch col names
    # print(str(col_name_list))
    proxylist = db.fetchall()

    for i in proxylist:
        txt = i[4]
        result = ' '.join(word for word in txt.split(' ') if not word.startswith('@'))
        if result is "":
            proxylist.remove(i)
        elif not has_num(result):
            proxylist.remove(i)
        else:
            sql = """ select shortcode, caption from post where id = ? """
            db.row_factory = sqlite3.Row
            db.execute(sql,(i[0],))
            d = db.fetchone()
            # print(str(d[1]))
            newlist.append([d[0],d[1],i[0],i[1],i[2],i[3],i[4]])
            # newlist.append(i[0],[i[1],i[2],i[3],i[4],d[0],d[1]])
            # proxylist[i].insert(d[0])
            # proxylist[i].insert(d[1])
        # print(str(db.fetchone()[0]))

with open('leads2.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(newlist)

print(str(newlist))
    # print(str(i[4]))

# con = sqlite3.connect(":memory:")
# con.row_factory = dict_factory
# cur = con.cursor()
# cur.execute("select 1 as a")
# print cur.fetchone()["a"]

# desc = cursor.description
# column_names = [col[0] for col in desc]
# data = [dict(itertools.izip(column_names, row))  
#         for row in cursor.fetchall()]

# print(str(proxylist))
