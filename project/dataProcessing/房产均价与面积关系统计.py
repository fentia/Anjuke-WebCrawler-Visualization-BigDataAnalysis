import re

input_josn = 'project\data\CleanData\cleanData.json'

data = []

with open(input_josn, 'r', encoding='utf-8') as infile:
        for line in infile:
            l = [] 
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
            # 均价
            avgPrice = pairs[9][1][:-3]
            # 面积
            area = pairs[2][1][:-1]
            # 所属小区
            community = pairs[6][1]
            l.append(avgPrice)
            l.append(area)  
            l.append(community)
            data.append(l)
print(data)
with open("project\dataVisualization\data\\avgPriceAndArea.json", 'w', encoding='utf-8') as outfile:
    for item in data:
        outfile.write('[' + item[1] + ',' + item[0] + ',"' + item[2] + '"]' + ',' + '\n')