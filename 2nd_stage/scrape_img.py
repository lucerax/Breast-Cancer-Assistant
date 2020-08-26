from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
import urllib3
import urllib
import pandas as pd
import requests
import os


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

browser = webdriver.Chrome(ChromeDriverManager().install())
#browser = webdriver.Chrome("chromedriver.exe")
#so far we have an instance of chromedriver that we use to access our page via get request





def scrape(iterations, url, category):
    link = url
    browser.get(link)
    try:
        for i in range(iterations):
            img_block = browser.find_element_by_id("image")
            img = browser.find_element_by_tag_name('img')
            print(img.text)
            img_url = browser.find_elements_by_xpath('//*[@id="image"]/div[2]/div/img')
            img_url = img_url[0].get_attribute('src')
            print(img_url)
            local_path = os.path.join('webpath', category)
            if not os.path.exists(local_path):
                os.mkdir(local_path)

            f = open(os.path.join(local_path, category + str(i) + '.jpg'),'wb')
            #f.write(urllib.request.urlopen(img_url).read())
            f.write(requests.get(img_url).content)
            f.close()
            next = img_block.find_element_by_xpath('/html/body/div/div[4]/div[1]/div[1]/div[3]/div[1]/table/tbody/tr/td[3]/a')
            next.click()

    except:
        print('check inputs for ', category)

def scrape2(base, start, end, category):
    try:
        for i in range(start, end + 1):
            url = base + '&n=' + str(i)
            browser.get(url)
            img_block = browser.find_element_by_id("image")
            img = browser.find_element_by_tag_name('img')
            print(img.text)
            img_url = browser.find_elements_by_xpath('//*[@id="image"]/div[2]/div/img')
            img_url = img_url[0].get_attribute('src')
            print(img_url)
            local_path = os.path.join('webpath', category)
            if not os.path.exists(local_path):
                os.mkdir(local_path)

            f = open(os.path.join(local_path, category + str(i) + '.jpg'),'wb')
            #f.write(urllib.request.urlopen(img_url).read())
            f.write(requests.get(img_url).content)
            f.close()
    except Exception as e:
        print("CHECK INPUT")
        print(e)


"""
scrape(10, 'https://www.webpathology.com/image.asp?case=289&n=1', 'DCIS')
scrape(18, 'https://www.webpathology.com/image.asp?case=291&n=1', 'LCIS')
scrape(2, 'https://www.webpathology.com/image.asp?case=302&n=1', 'metaplastic')
scrape(2, 'https://www.webpathology.com/image.asp?case=293&n=1', 'papillary')
scrape(22, 'https://www.webpathology.com/image.asp?case=290&n=30', 'IDC')

scrape(33, 'https://www.webpathology.com/image.asp?case=292&n=1', 'ILC')
scrape(21, 'https://www.webpathology.com/image.asp?case=292&n=35', 'ILC_1')

scrape(2, 'https://www.webpathology.com/image.asp?case=296&n=1', 'tubular')
scrape(9, 'https://www.webpathology.com/image.asp?case=296&n=4', 'tubular_1')

scrape(26, 'https://www.webpathology.com/image.asp?case=297&n=5', 'mucinous')
scrape(22, 'https://www.webpathology.com/image.asp?case=298&n=3', 'medullary')
scrape(23, 'https://www.webpathology.com/image.asp?case=299&n=2', 'apocrine')
scrape(12, 'https://www.webpathology.com/image.asp?case=300&n=1', 'secretory')
"""
#scrape(19, 'https://www.webpathology.com/image.asp?case=324&n=1', 'adenoid_cystic')
#scrape(11, 'https://www.webpathology.com/image.asp?case=324&n=35', 'adenoid_cystic_2')
#scrape(8, 'https://www.webpathology.com/image.asp?case=303&n=8', 'paget')
#scrape(10, 'https://www.webpathology.com/image.asp?case=651&n=4', 'other')

"""
scrape2('https://www.webpathology.com/image.asp?case=311', 19, 25, 'phyllodes_malignant')
scrape2('https://www.webpathology.com/image.asp?case=316', 1, 1, 'liposarcoma')
scrape2('https://www.webpathology.com/image.asp?case=319', 1, 2, 'chondrosarcoma')
scrape2('https://www.webpathology.com/image.asp?case=317', 1, 2, 'dermatofibrosarcoma')
scrape2('https://www.webpathology.com/image.asp?case=320', 3, 6, 'angiosarcoma_benign')
scrape2('https://www.webpathology.com/image.asp?case=320', 9, 10, 'angiosarcoma_malignant')
scrape2('https://www.webpathology.com/image.asp?case=276', 5, 13, 'adenoma')
"""


scrape2('https://www.webpathology.com/image.asp?case=293', 1, 18, 'papillary_carcinoma')
