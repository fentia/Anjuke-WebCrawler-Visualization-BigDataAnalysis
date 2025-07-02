import re

input_josn = 'project\data\CleanData\cleanData.json'

regionalAveragePriceDit = {}

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
            # 区域
            houseType = pairs[7][1][0:2]
            # 均价
            averagePrice = float(pairs[9][1][:-3])  # 去掉单位“元/平米”
            if houseType in regionalAveragePriceDit:
                regionalAveragePriceDit[houseType].append(averagePrice)
            else:
                regionalAveragePriceDit[houseType] = [averagePrice]
# 计算区域均价
for key, value in regionalAveragePriceDit.items():
    regionalAveragePriceDit[key] = sum(value) / len(value)
# 输出区域均价统计
print('区域均价统计：')
regionalL = []
priceL = []
for key, value in regionalAveragePriceDit.items():
    print(f'{key}：{value:.2f}元/平米')
    regionalL.append(key)
    priceL.append(f'{value:.2f}')
print('区域均价列表：')
print(regionalL)
print(priceL)
# ['高坪', '顺庆', '嘉陵', '阆中', '营山', '南部', '蓬安', '西充', '仪陇']
# ['5622.25', '6842.08', '5035.69', '4617.09', '4619.83', '5114.64', '4947.58', '4153.88', '5127.61']