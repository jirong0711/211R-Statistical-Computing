import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import warnings
warnings.filterwarnings(action='ignore')
import re
from ckonlpy.tag import Postprocessor
from ckonlpy.utils import load_wordset

def KBS(url):
    request = requests.get(url)
    bs = BeautifulSoup(request.text, "html.parser")
    article = bs.find('div', {'id':"cont_newstext"}).text.strip()
    return(article)
def MBC(url):
    driver.get(url)
    article = driver.find_element_by_css_selector(
        '#content > div > section.wrap_article > article > div.news_cont > div.news_txt').text.strip()
    return(article)
def SBS(url):
    request = requests.get(url)
    bs = BeautifulSoup(request.text, "html.parser")
    div = bs.find('div', class_='text_area').text.strip()
    return(div)


##Data Scraping
list = pd.read_excel('data/broadcast_2021_1.xlsx').iloc[:,[1,2,17]]
articles = [0 for i in range(len(list))]
list["content"] = 0
driver = webdriver.Chrome('/Users/jihyunlee/Downloads/chromedriver-2')
for i in range(len(list)):
    if list.iloc[i,1] == 'KBS':
        try:
            list.iloc[i,-1] = KBS(list.iloc[i,2])
        except:
            list.iloc[i,-1] = ""
    elif list.iloc[i,1] == 'SBS':
        try:
            list.iloc[i,-1] = SBS(list.iloc[i,2])
        except:
            list.iloc[i,-1] = ""
    elif list.iloc[i,1] == 'MBC':
        try:
            list.iloc[i,-1] = MBC(list.iloc[i,2])
        except:
            list.iloc[i,-1] = ""


##Data Preprocessing
broad11 = broad1.copy()
broad11 = broad11.rename({'Unnamed: 0':'index'}, axis=1)

#dictionary�� �ܾ� �߰�
twitter.add_dictionary('Ȯ����', 'Noun')
twitter.add_dictionary('�Կ�����', 'Noun')
twitter.add_dictionary('������', 'Noun')
twitter.add_dictionary('Ű����', 'Noun')
twitter.add_dictionary('������', 'Noun')

#�ҿ�� ����Ʈ �ҷ�����
stopwords = load_wordset('stopwords.txt')

postprocessor = Postprocessor(
    base_tagger = twitter,
    stopwords = stopwords, # �ҿ�� ����
    passtags = 'Noun', # ��縸 ����
)

documents = []
for txt in text:
    txt = re.sub(r'[0-9]+', '', txt) #���� ����
    txt = re.sub('[-=+,#/\?:^$.@*\"��~&%��!��\\��|\(\)\[\]\<\>`\'����]', '', txt)  #Ư������ ����
    words = postprocessor.pos(txt)
    new = ""
    for word in words:
        if len(word[0]) > 1: #�α��� �̻� �ܾ� ����
            new += " " + word[0]
    documents.append(new)

#broad11["content"] = documents
#broad11.to_csv("broad11.csv")