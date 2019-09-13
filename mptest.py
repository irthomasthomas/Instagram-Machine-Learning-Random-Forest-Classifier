from multiprocessing import Process, Queue, Manager, Pool
import random
import sys
import os
import json
import time

result = []

def load_json_posts_file_dir(path,posts):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            posts.append(json.loads(f))
            # print("posts len: " + str(len(posts)))
            # print(str(json.loads(f)))

def load_json_page(file,posts):
    with open(file, 'r') as fp:
        f = fp.read()
        posts.append(json.loads(f))
    # print("posts len: " + str(len(posts)))

def load_json_from_fileobj(file):
    global result
    jfile = json.loads(file)
    result.append(jfile)

def fill_file_list(file,files):
    with open(file, 'r') as fp:
        files.append(fp.read())

def single_thread_test(posts):
    start = time.time()
    # posts = []
    load_json_posts_file_dir(path, posts)

    print("single: end posts len: " + str(len(posts)))

    end = time.time()
    print(end - start)

def single_read():
    start = time.time()
    pool = Pool(2)

    # with Manager() as manager:
        # posts = manager.list()
    jobs = []
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
        p = Process(target=load_json_from_fileobj, args=(f,))
        jobs.append(p)
        p.start()
        p.join()
    print("single: end jobs[] len: " + str(len(jobs)))
    end = time.time()
    print(end - start)


def multi_threads_test():
    start = time.time()
    with Manager() as manager:
        posts = manager.list()
        # queue = Queue()
        for file in os.scandir(path):
            processes = [Process(target=load_json_page, args=(file,posts))]
        
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        print("multi: end posts len: " + str(len(posts)))

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    if len(sys.argv) >= 1:
        path=sys.argv[1]
    posts = []
    # multi_threads_test()
    single_thread_test(posts)
    print("posts: " + str(len(posts)))
    # single_read()
    # print("single: end result[] len: " + str(len(result)))

