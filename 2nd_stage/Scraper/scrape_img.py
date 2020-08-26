from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
import urllib3
import pandas as pd
from sqlalchemy import create_engine
from xlrd import open_workbook
from xlutils.copy import copy
import os


browser = webdriver.Chrome("chromedriver.exe")

#so far we have an instance of chromedriver that we use to access our page via get request





def scrape(iterations, url, category):
    link = url
    browser.get(link)
    for i in iterations:
        img_block = browser.find_element_by_id("image")
        img = img_block.find_element_by_tag_name('img')
        img_url = img.get_attribute('src')
        print(img_url)
        local_path = os.path.join('webpath', category)
        if not os.path.exists(local_path):
            os.mkdir(local_path)

        f = open(local_path,'wb')
        f.write(urllib.urlopen('img_url').read())
        f.close()
        next = img_block.find_element_by_xpath('/html/body/div/div[4]/div[1]/div[1]/div[3]/div[1]/table/tbody/tr/td[3]/a')
        next.click()

scrape(2, 'https://www.webpathology.com/image.asp?case=302&n=1', 'metaplastic')

book = open_workbook("policedepts.xlsx")
sheet = book.sheet_by_index(0)
copy_book = copy(book) # a writable copy (I can't read values out of this, only write to it)
copy_sheet = copy_book.get_sheet(0) # the sheet to write to within the writable copy

depts = []
results = []
for row in range(328):
    depts.append(sheet.cell(row,0).value)
for index, dept in enumerate(depts):
    result = scrape(dept)
    copy_sheet.write(index, 1, result)

copy_book.save('dept_out.xls')
