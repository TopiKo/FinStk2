from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import time
import datetime
import os

def get_stored():
    vals = []
    for file in os.listdir("data/"):
        if file.endswith(".pkl"):
            vals.append(file.split('.pkl')[0].split('_')[-1])
    return vals

def get_data(days, sleep_t = 1):

    driver = webdriver.Firefox()
    driver.get("https://www.kauppalehti.fi/5/i/porssi/porssikurssit/kurssihistoria.jsp")
    [t1, t2, t3] = .5*np.random.random(3)
    stored = get_stored()

    for day in days:
        if day not in stored:
            try:
                time.sleep(sleep_t + t1)
                elem = driver.find_element_by_id('kpv')
                time.sleep(sleep_t + t2)
                elem.clear()
                time.sleep(sleep_t + t1)
                elem.send_keys(day)
                time.sleep(sleep_t + t2)

                elem.send_keys(Keys.RETURN)
                time.sleep(sleep_t*4 + t3)

                date = driver.find_elements_by_tag_name('h3')

                print(date[-1].text.split(' ')[-1], day)
                if date[-1].text.split(' ')[-1] == day:
                    data_strs = driver.find_elements_by_tag_name('tr')
                    df = get_df(data_strs)
                    df.to_pickle('data/market_data_{:s}.pkl'.format(day))
                    print('alles good {:s}'.format(day), df.shape)
            except Exception as e:
                print('KAAKAAKKAAK {:s}'.format(day))
                print(e)
            print()

def get_df(elems):
    colls = ['offer_end', 'offer_buy', 'offer_sell',
             'sales_low', 'sales_high', 'change_Me']

    df = pd.DataFrame(columns = colls)

    for elem in elems:
        data = elem.text
        if len(data) != 0:
            if data.split(' ')[-1] == 'EUR':
                nums = data.split(' ')[-7:-1]
                company = data.split(' ')[:-7]
                company_n = ''
                if len(company) > 1:
                    for i in range(len(company)):
                        company_n += company[i] + ' '
                    company_n.strip()
                else:
                    company_n = company[0]
                num_np = np.zeros(6)
                for i, num in enumerate(nums):
                    try:
                        num_np[i] = float(num)
                    except (ValueError, TypeError):
                        num_np[i] = np.nan
                df.loc[company_n] = num_np

    return df


numdays = 1000 #365*15
base = datetime.datetime.today()
days = [base - datetime.timedelta(days=x) for x in range(1, numdays)]

weekdays = []
for day in days:
    if day.weekday() in range(5):
        weekdays.append(day.strftime("%d.%m.%Y") )

#days = ['19.8.2017', '20.8.2017', '21.8.2017', '22.8.2017', '23.8.2017', '24.8.2017']
get_data(weekdays, 1.5)
