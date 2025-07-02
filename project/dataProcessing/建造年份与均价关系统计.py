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
            # 均价
            price = pairs[9][1][:-3]
            if creatYear not in yearDit:
                yearDit[creatYear] = []
            yearDit[creatYear].append(float(price))
# 计算均价
for year in yearDit:
    yearDit[year] = sum(yearDit[year]) / len(yearDit[year])
yearDit = sorted(yearDit.items(), key=lambda x: x[0])  # 按年份排序
# 输出结果
yearNameL = []
yearPriceL = []
for year, price in yearDit:
    print(f"{year}: {price}元/平米")
    yearNameL.append(year)
    yearPriceL.append(price)
print(yearNameL)
print(yearPriceL)
# ['1992', '1993', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
# ['3311.50', '5767.00', '4733.55', '5851.00', '4237.00', '3258.12', '3692.10', '4159.02', '3554.12', '3460.35', '3717.07', '3273.13', '3324.67', '3742.91', '3553.13', '3834.99', '4340.17', '4749.09', '4919.82', '5255.25', '5255.61', '5450.33', '5331.75', '5466.88', '5561.40', '5732.96', '5670.22', '5693.17', '5595.42', '5671.56', '9827.17', '5712.42', '5355.95']