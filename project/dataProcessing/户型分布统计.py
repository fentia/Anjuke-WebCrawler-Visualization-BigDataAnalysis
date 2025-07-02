import re

input_josn = 'project\data\CleanData\cleanData.json'

houseTypeDit = {'3室2厅2卫':0, '4室2厅2卫':0, '3室1厅2卫':0, '2室2厅1卫':0, '其他':0}

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
            # 户型
            houseType = eval(pairs[1][1])

