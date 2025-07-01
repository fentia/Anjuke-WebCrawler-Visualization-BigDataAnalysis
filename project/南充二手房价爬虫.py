import json
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def get_data():
    option = webdriver.ChromeOptions()
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    option.add_experimental_option('detach', True)
    option.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=option)

    data_list = []

    y_list = ['1', '2']
    for y in y_list:
        for p in range(1, 2):

            url = 'https://nanchong.anjuke.com/sale/p' + str(p) + '-y' + y + '/?from=fangjia'
            print(url)
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, 'lxml')
            soup_list = soup.select('.property')

            try:
                page_1 = soup.select('.page .active')[0].text.replace('\n', '').strip()
                if page_1 == '1' and p != 1:
                    break
            except:
                try:
                    end_text = soup.select('.list-guess-title')[0].text
                    if '猜你喜欢' in end_text:
                        break
                except:
                    pass

            for sl in soup_list:
                data = {}

                data['标题'] = sl.select('.property-content-title-name')[0].text
                data['户型'] = sl.select('.property-content-info-text.property-content-info-attribute')[0].text
                data['面积'] = ''
                data['方位'] = ''
                data['楼层'] = ''
                data['时间'] = ''
                dt_list = sl.select('.property-content-info-text')
                for dl in dt_list:
                    if '㎡' in dl.text:
                        data['面积'] = dl.text.replace('\n','').strip()
                    if '东' in dl.text or '南' in dl.text or '西' in dl.text or '北' in dl.text:
                        data['方位'] = dl.text.replace('\n','').strip()
                    if '层' in dl.text:
                        data['楼层'] = dl.text.replace('\n','').strip()
                    if '建造' in dl.text:
                        data['时间'] = dl.text.replace('\n','').strip()
                data['所属小区'] = sl.select('.property-content-info-comm-name')[0].text
                data['所属区域'] = sl.select('.property-content-info-comm-address')[0].text


                # data['总价'] = sl.select('.property-price-total')[0].text

                total_price_elements = sl.select('.property-price-total')
                if total_price_elements:
                    data['总价'] = total_price_elements[0].text
                else:
                    data['总价'] = 'N/A'

                # data['均价'] = sl.select('.property-price-average')[0].text
                average_price_elements = sl.select('.property-price-average')
                if average_price_elements:
                    data['均价'] = average_price_elements[0].text
                else:
                    data['均价'] = 'N/A'
                data['房龄'] = '2年内'
                if y == '2':
                    data['房龄'] = '2-5年'

                print(data)
                with open('project/data.json', 'a', encoding='utf8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + ',\n')
                data_list.append(data)
            print(p)

    df = pd.DataFrame(data_list)
    df.to_excel('project/数据.xlsx', index=False)


get_data()