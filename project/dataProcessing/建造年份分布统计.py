import re

input_josn = 'project\data\CleanData\cleanData.json'

yearDit = {}

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
            # 建造时间
            creatYear = pairs[5][1][0:4]
            if creatYear in yearDit:
                yearDit[creatYear] += 1 
            else:
                yearDit[creatYear] = 1
# 排序
yearDit = dict(sorted(yearDit.items(), key=lambda item: item[0]))

# 输出建造年份分布统计
print('建造年份分布统计：')
yearL = []
countL = []
for key, value in yearDit.items():
    print(f'{key}年：{value}套')
    yearL.append(f'{key}年')
    countL.append(value)

print(yearL)
print(countL)