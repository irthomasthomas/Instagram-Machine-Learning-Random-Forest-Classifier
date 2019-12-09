from InstaLoaderTommy import InstaloaderTommy

# session = requests.session()
scraper = InstaloaderTommy()

with scraper:
    for count in scraper.get_hashtag_posts(
        'headyart', resume=True):
        print(f'Count: {count}')
