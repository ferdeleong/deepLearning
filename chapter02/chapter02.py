
key = 'XXX'
key = os.environ['AZURE_SEARCH_KEY']


search_images_bing

results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)

dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
im = Image.open(dest)
im.to_thumb(128,128)