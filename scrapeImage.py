import io
from urllib.request import urlopen

from PIL import Image
from bs4 import BeautifulSoup as bSoup
from requests import get


def getthing(titleid):
    url = "https://www.imdb.com/title/" + titleid
    html = get(url)
    soup = bSoup(html.text, 'html.parser')
    poster_contain = soup.find_all('div', class_='poster')
    poster = poster_contain[0]
    imageurl = poster.a.img[
        'src']  # Example URL: https://m.media-amazon.com/images/M/MV5BMTg5NzY0MzA2MV5BMl5BanBnXkFtZTYwNDc3NTc2._V1_SX225_CR0,0,225,333_AL_.jpg
    split = str(imageurl).split(",")
    original2 = split[2]
    split[0] = split[0].replace(original2, str(int(original2) * 2))
    split[2] = str(int(split[2]) * 2)
    split[3] = str(int(split[3][:split[3].find("_")]) * 2) + split[3][split[3].find(
        "_"):]  # WHAT THE HELL. This line gets split[3], doubles the number inside it and sets the string back
    finalstr = ",".join(split)
    print(finalstr)
    return finalstr


fd = urlopen(getthing("tt0317219"))
image_file = io.BytesIO(fd.read())
im = Image.open(image_file)
im.show()
