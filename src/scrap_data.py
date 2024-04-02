'''
Add the scrapping code here

output
-> data/interim : table.csv (provessed csv file of drugs-
-> data/raw: some pdf text related to reglementation of drugs of one cluster that we'll use as examples
'''
import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')

import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller
from tqdm import tqdm

# setup chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# set path to chromedriver as per your configuration
chromedriver_autoinstaller.install()

# set up the webdriver
driver = webdriver.Chrome(options=chrome_options)

def urls_opendata(path):
    # Read the Excel file into a Pandas DataFrame
    df = pd.read_excel(path)

    # Process the DataFrame
    df1 = df.iloc[7:]
    df1.columns = df1.iloc[0]
    df1 = df1[1:]
    df1.reset_index(drop=True, inplace=True)
    df1 = df1.dropna(how='all').dropna(how='all', axis=1)
    df2 = df1[df1['Authorisation status'] == 'Authorised']

    # Get the URLs from the DataFrame
    urls = df2['URL']

    return df2, urls


def get_data(urls):
    # Iterate over the URLs and scrape the data
    descriptions = []
    driver.get(urls[1])
    driver.maximize_window()
    time.sleep(2)
    # Accept all cookies
    #l=driver.find_element("link text", "Accept all cookies")
    #l.click()
    #time.sleep(2)

        # close button
    #b =driver.find_element('xpath', '//*[@id="cookie-consent-banner"]/div/div/div[2]/button')
    #b.click()

    for i in urls:
        driver.get(i)

        name = driver.find_element('xpath', '//div[@class="heading-title"]/h1/span').text
        active_substance = driver.find_element('xpath', '//*[@id="block-system-main-block"]/article/div[1]/div[2]/div/div[3]/div[1]/dl/dd[2]').text
        common_name = driver.find_element('xpath', '//*[@id="block-system-main-block"]/article/div[1]/div[2]/div/div[3]/div[1]/dl/dd[3]').text
        therap_area = driver.find_element('xpath', '//*[@id="block-system-main-block"]/article/div[1]/div[2]/div/div[3]/div[1]/dl/dd[4]').text
        #therapeutic_indication = driver.find_elements('xpath', '//*[@id="block-system-main-block"]/article/div[1]/div[2]/div/div[3]/div[3]')[0].text # + ' ' + driver.find_elements('xpath', '//*[@id="block-system-main-block"]/article/div[1]/div[2]/div/div[3]/div[3]')[1].text


        # Medicament usage
        usa = driver.find_element('xpath', "//span[contains(text(),'used') and contains(text(),'?')]")
        driver.execute_script('arguments[0].scrollIntoView(true)', usa)
        time.sleep(3)
        usa.find_element('xpath', '../..').click()
        usa.find_elements('xpath', '../../..')[-1].text
        usa.find_elements('xpath', '../../..')[-1].text
        usage = usa.find_elements('xpath', '../../..')[-1].text
        time.sleep(3)


        # Medicament risks
        usb = driver.find_element('xpath', "//span[contains(text(),'risk') or contains(text(),'side effects') or contains(text(),'side-effects') and contains(text(),'?')]")
        driver.execute_script('arguments[0].scrollIntoView(true)', usb)
        time.sleep(3)
        usb.find_element('xpath', '../..').click()
        usb.find_elements('xpath', '../../..')[0].text
        usb.find_elements('xpath', '../../..')[0].text
        risk = usb.find_elements('xpath', '../../..')[0].text

        prodJ = {
            'name' : name,
            'active_substance' : active_substance,
            'common_name' : common_name,
            'therap_area' : therap_area,
            # 'therapeutic_indication' : therapeutic_indication,
            'usage': usage,
            'risk' : risk,
            'medicamentLink': i
        }
        descriptions.append(prodJ)

    return descriptions

def data_from_EMA(path):
    df2, urls = urls_opendata(path)
    df = pd.DataFrame(get_data(urls[0:10]))
    ndf = pd.merge(df, df2, left_on='name', right_on='Medicine name', how='outer')

    # Save the processed DataFrame to a CSV file
    ndf.to_csv('/content/drive/MyDrive/stage/hh/Medical_Reglementation/data/raw/data_scraped.csv', index=False)
    return ndf
if __name__ == "__main__":
  path = "./data/interim/medicines_output_european_public_assessment_reports_en.xlsx"
  df = data_from_EMA(path)
  df.head()
