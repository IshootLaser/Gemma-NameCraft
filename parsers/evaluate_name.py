from settings import name_eval_url, logger
import logging
from bs4 import BeautifulSoup
import requests
from collections import OrderedDict


def get_hour(hour: int):
    cbo_hour = "0-子时"
    if int(hour) in [0, 23]:
        cbo_hour = "%s-子时" % hour
    elif int(hour) in [1, 2]:
        cbo_hour = "%s-丑时" % hour
    elif int(hour) in [3, 4]:
        cbo_hour = "%s-寅时" % hour
    elif int(hour) in [5, 6]:
        cbo_hour = "%s-卯时" % hour
    elif int(hour) in [7, 8]:
        cbo_hour = "%s-辰时" % hour
    elif int(hour) in [9, 10]:
        cbo_hour = "%s-巳时" % hour
    elif int(hour) in [11, 12]:
        cbo_hour = "%s-午时" % hour
    elif int(hour) in [13, 14]:
        cbo_hour = "%s-未时" % hour
    elif int(hour) in [15, 16]:
        cbo_hour = "%s-申时" % hour
    elif int(hour) in [17, 18]:
        cbo_hour = "%s-酉时" % hour
    elif int(hour) in [19, 20]:
        cbo_hour = "%s-戌时" % hour
    elif int(hour) in [21, 22]:
        cbo_hour = "%s-亥时" % hour
    return cbo_hour


def prepare_payload(
        _last_name: str = '王',
        _first_name: str = '小狗',
        _year: int = 2024,
        _month: int = 12,
        _day: int = 31,
        _hour: int = 24,
        _minute: int = 59,
        _province: str = '福建',
        _city: str = '厦门',
        is_boy=True
):
    data = {
        'isbz': 1,
        'txtName': _last_name,
        'name': _first_name,
        'rdoSex': 1 if is_boy else 0,
        'data_type': 0,
        'cboYear': _year,
        'cboDay': _day,
        'cboMonth': _month,
        'cboHour': get_hour(_hour),
        'cboMinute': f'{_minute}分',
        'pid': _province,
        'cid': _city,
        'zty': 0
    }
    return data


def get_name_eval(payload: dict):
    eval_result = OrderedDict()
    try:
        r = requests.post(name_eval_url, data=payload)
        if r.status_code != 200:
            raise Exception(f'Failed to get name evaluation result from {name_eval_url}. Status code: {r.status_code}')
        soup = BeautifulSoup(r.content, 'html.parser')
        _ = soup.find('span', class_='df_1 left')
        name_score = float(_.text.split(':')[1].strip())
        _ = soup.find('span', class_='df_1 right')
        bazi_score = float(_.text.split(':')[1].strip())
        eval_result = {
            'name_score': name_score, 'bazi_score': bazi_score
        }
        bazi_box = soup.find('ul', class_='bazi_box')
        replace_chars = (
            ('>', ''),
            ('\xa0', ' ')
        )
        for li in bazi_box.find_all('li'):
            label = li.find('strong').text[:-1]  # Remove the colon from the label
            value = li.text.replace(label + ':', '').strip()
            for a, b in replace_chars:
                value = value.replace(a, b)
            eval_result[label] = value
        wuxing_mgl10 = soup.find('div', class_='sm_wuxing mgl10')
        eval_result['wuxing'] = wuxing_mgl10.find('strong').text
        eval_result['ji_chu_yun'] = soup.select(
            'body > div:nth-child(6) > div > div.qml.left.mgb10 > div.sm_body > div:nth-child(23)'
        )[0].text
        eval_result['cheng_gong_yun'] = soup.select(
            'body > div:nth-child(6) > div > div.qml.left.mgb10 > div.sm_body > div:nth-child(24)'
        )[0].text
        eval_result['she_jiao_yun'] = soup.select(
            'body > div:nth-child(6) > div > div.qml.left.mgb10 > div.sm_body > div:nth-child(25)'
        )[0].text
        eval_result['personal_trait'] = soup.select(
            'body > div:nth-child(6) > div > div.qml.left.mgb10 > div.sm_body > div:nth-child(26)'
        )[0].text
        chart = soup.select(
                'body > div:nth-child(6) > div > div.qml.left.mgb10 > div.sm_body > ul.xingmingzili'
        )[0]
        for li in chart.find_all('li', class_='xmzili2'):
            _ = li.text
            eval_result[_[:3]] = _
        eval_result['zong_ge'] = chart.find('li', class_='geshu').text
    except Exception as e:
        msg = f'Failed to get name evaluation result from {name_eval_url}. Error: {e}'
        logger.log(logging.ERROR, msg)
    return eval_result


if __name__ == '__main__':
    # a simple test
    from pprint import pformat
    _payload = prepare_payload(_first_name='大拿')
    _eval_result = get_name_eval(_payload)
    print(pformat(_eval_result))
