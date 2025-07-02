import re

input_josn = 'project\data\CleanData\cleanData.json'

communityDit = {}

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
            # 小区
            community = pairs[6][1]
            if community not in communityDit:
                communityDit[community] = 1
            else:
                communityDit[community] += 1
communityDit = sorted(communityDit.items(), key=lambda x: x[1], reverse=True)
# 输出结果
communityNameL = []
communityCountL = [] 
for community, count in communityDit[0:10]:  # 只输出前10个小区
    print(f"{community}: {count}套房源")
    communityNameL.append(community)
    communityCountL.append(count)
print(communityNameL)
print(communityCountL)
# ['蓝光香江国际', '阳光江山公园城', '碧桂园原树缇香', '铁投锦华府', '明宇帝壹家西区', '春风玫瑰园', '江东首席', '碧桂园天樾', '华邦天誉', '泰合尚渡']
# [129, 93, 74, 69, 69, 62, 61, 60, 59, 59]
