import urllib.request# url request
import re            # regular expression
import os            # dirs
import time
import requests
import sys
from keras.utils import get_file
# Parsing HTML
from bs4 import BeautifulSoup

"""## Searching for Wikipedia Download
To start, we make a request to the [Wikimedia dumps](https://dumps.wikimedia.org/) of Wikipedia. We'll search through `enwiki` which has the [English language dumps](https://dumps.wikimedia.org/enwiki/) of wikipedia. This first request finds the available recent dumps and lists them. (A dump is a snapshot of all the existing information from a database).
"""

base_url = 'http://www.kunstderfuge.com/beethoven/klavier.htm#Harpsichord'
index = requests.get(base_url).text
soup_index = BeautifulSoup(index, 'html.parser')

# Find the links that are dates of dumps
dumps = [a['href'] for a in soup_index.find_all('a') if 
         a.has_attr('href')]
dumps

"""The next line of code finds the html content of the page for the dump made on the first of September. If there is a more recent version available, feel free to use that instead!"""

dump_url = base_url + '20200201/'

# Retrieve the html
dump_html = requests.get(dump_url).text
dump_html[:10]


base_url = "http://www.kunstderfuge.com/-/midi.asp?file="
work_name = "beethoven/symphony_1_1_(c)lucarelli.mid"
download_url = base_url + work_name

directory = os.path.abspath("/Users/huanzhang/01Acdemics/College/2020Spring/10701_IntroToML/Data/")
path = directory + "/" + work_name

get_file(path, download_url)

# # Iterate through each file
# for file in files_to_download:
#     path = keras_home + file
    
#     # Check to see if the path exists (if the file is already downloaded)
#     if not os.path.exists(path):
#         print('Downloading')
#         # If not, download the file
#         print(get_file(file, dump_url))



'''
url 内容网址（包含有哪些数据的内容）
dataloc 下载网址（数据存放处）
pattern 正则化的匹配关键词
Directory 下载目录
'''
# def BatchDownload(url, dataloc,pattern,Directory):
def BatchDownload(composer):
    url = 'https://www.kunstderfuge.com/ravel.html'
    dataloc = 'https://www.midiworld.com/midis/other/'
    pattern = '(' + composer + '(\S*).mid)'
    Directory = ''
    
    # 创建一个composer folder
    if not os.path.exists(composer):
        print('Creating directory ' + composer)
        os.makedirs(composer)

    # 拉动请求，模拟成浏览器去访问网站->跳过反爬虫机制
    headers = {'User-Agent', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}
    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    
    # 获取网页内容
    content = opener.open(url).read().decode('utf8')
    
    print(content)
    return

    # 构造正则表达式，从content中匹配关键词pattern
    raw_hrefs = re.findall(pattern, content, 0)
    
    # set函数消除重复元素
    hset = set(raw_hrefs)
    

    # 下载链接
    for href in hset:
        # 之所以if else 是为了区别只有一个链接的特别情况
        if(len(hset)>1):
            link = dataloc + href[0]
            filename = os.path.join(Directory, href[0])
            print("正在下载",filename)
            urllib.request.urlretrieve(link, filename)
            print("成功下载！")
        else:
            link = dataloc +href
            filename = os.path.join(Directory, href)
            print("正在下载",filename)
            urllib.request.urlretrieve(link, filename)
            print("成功下载！")
            
        # 无sleep间隔，网站认定这种行为是攻击，反反爬虫
        time.sleep(3)

# https://www.midiworld.com/midis/other/bach/bwv806.mid
# BatchDownload('https://www.midiworld.com/bach.html',
# 			  'https://www.midiworld.com/midis/other/',
#              '(chopin(\S*).mid)',
#              '')
# BatchDownload('byrd')
        