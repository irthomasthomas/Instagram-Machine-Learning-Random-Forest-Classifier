import redis
from sys import argv
# delete
# tag:posts
# post:tags
# tags:out
# TODO: root:tag
# TODO: page:cursor
# TODO: sketch:all
rdb = redis.Redis(decode_responses=True)
input = argv[1]
startkey = f'tag:posts:{input}'


# Get all posts under tag:posts: startkey
posts = rdb.smembers(startkey) # 10 #sweatshirts
print(f'posts: {len(posts)}')
for postId in posts:
    key = f'tag:posts:{postId}'
    rdb.delete(key)

# Get all post:tags for all posts in startkey
hashtags = set()
for postId in posts:
    key = f'post:tags:{postId}'
    for tag in rdb.smembers(key):
        hashtags.add(tag)
        rdb.delete(f'tags:out:{tag}')

print(f'all tags set: {len(hashtags)}')

# get all tag:posts for tags in hashtags
related_posts = set()
for tag in hashtags:
    rdb.delete(f'tags:out:{tag}')
    for post in rdb.smembers(f'tag:posts:{tag}'):
        # print(f'related post {post}')
        related_posts.add(post)

related_tags = set()
for postId in related_posts:
    key = f'post:tags:{postId}'
    for tag in rdb.smembers(key):
        rdb.delete(f'tags:out:{tag}')
        sketch = f'sketch:all:{tag}'
        rdb.delete(sketch)
        related_tags.add(tag)

next_posts = set()
for tag in related_tags:
    rdb.delete(f'tags:out:{tag}')
    key = f'tag:posts:{tag}'
    for post in rdb.smembers(key):
        next_posts.add(post)
    rdb.delete(key)

for post in next_posts:
    key = f'post:tags:{post}'
    rdb.delete(key)
print(f'related tags: {len(related_tags)}')
print(f'next_posts: {len(next_posts)}')
print(f'Delete {startkey}')
print(f'Delete {len(related_posts)} tag:posts:')
for postId in related_posts:
    key = f'post:tags:{postId}'
    rdb.delete(key)

# print(related_posts.pop())
# print(f'Delete {len(hashtags)} post:tags')
# for tag in hashtags:
#     key = f'tag:posts:{tag}'
#     rdb.delete(key)
# print('finished')

rdb.delete(startkey)
# sweepstakes
# 
# print(hashtags.pop())
# print(f'Delete {len(related_tags)} post:tags')
# print(related_tags.pop())
# for p in related_posts:
#     key = f'post:tags:{p}'
#     print(f'deleting related_post: {key}')
#     # rdb.delete(key)

# print(f'posts size: {len(posts)}')
# for p in posts:
#     key = f'post:tags:{p}'
    # print(f'deleting: {key}')
    # rdb.delete(key)

# rdb.delete(startkey)

# print(hashtags)
# for postId in posts:
#     rdb.delete(f'post:tags:{postId}')

