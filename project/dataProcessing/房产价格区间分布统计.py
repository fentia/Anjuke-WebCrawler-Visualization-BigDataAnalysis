import re

input_josn = 'project\data\CleanData\cleanData.json'

totalPrice = {'30万以下':0, '30-40万':0, '40-50万':0, '50-60万':0, '60-80万':0, '80-100万':0, '100万以上':0}

with open(input_josn, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            # 提取 "键": "值" 对
            '''
            # [('标题', ''), ('户型', ''), ('面积', ''), ('方位', ''), 
            # ('楼层', ''), ('时间', ''), ('所属小区', ''), ('所属区域', ''), 
            # ('总价', ''), ('均价', ''), ('房龄', '')]
            '''
            pairs = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', line)
            # 将价格转换为整数，单位为元
            price = eval(pairs[8][1][0:-1])* 10000
            if price < 300000:
                totalPrice['30万以下'] += 1
            elif 300000 <= price < 400000:
                totalPrice['30-40万'] += 1
            elif 400000 <= price < 500000:
                totalPrice['40-50万'] += 1
            elif 500000 <= price < 600000:
                totalPrice['50-60万'] += 1
            elif 600000 <= price < 800000:
                totalPrice['60-80万'] += 1
            elif 800000 <= price < 1000000:
                totalPrice['80-100万'] += 1
            elif price >= 1000000:
                totalPrice['100万以上'] += 1
print('房产价格区间分布统计：')
# 价格列表
l = []
for key, value in totalPrice.items():
    l.append(value)
    print(f'{key}: {value}套')
print(l)
# Output: 房产价格区间分布统计：
# [446, 1072, 1686, 2248, 1943, 501, 338]