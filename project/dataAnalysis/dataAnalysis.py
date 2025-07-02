import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 加载数据
df = pd.read_excel('project\data\CleanData\cleanData.xlsx')

# 2. 数据清理与转换
# 处理缺失值和重复值
df = df.dropna().drop_duplicates()

# 编码分类变量
le = LabelEncoder()

# 户型
df['房间数'] = df['户型'].str.extract(r'(\d+)室')[0].astype(int)
df['客厅数'] = df['户型'].str.extract(r'(\d+)厅')[0].astype(int)
df['卫生间数'] = df['户型'].str.extract(r'(\d+)卫')[0].astype(int)

# 面积
df['面积'] = df['面积'].str.replace('㎡', '').astype(float)

# 总价
df['总价'] = df['总价'].str.replace('万', '').astype(float) * 10000

# 均价
df['均价'] = df['均价'].str.replace('元/㎡', '').astype(float)

# 建造年份
df['建造年份'] = df['时间'].str.extract(r'(\d{4})年建造')[0].astype(int)

# 楼层 & 楼层编码
df['楼层类别'] = df['楼层'].str.extract(r'([低中高]层)')[0]
df['楼层编码'] = le.fit_transform(df['楼层类别'])
'''
# 楼层类别和编码的对应关系
   楼层类别  楼层编码
0    高层     2
2    中层     0
4    低层     1
55  NaN     3
'''

# 总楼层
df['总楼层'] = df['楼层'].str.extract(r'共(\d+)层')[0].astype(float)

# 区域 & 区域编码
nanchong_districts = ['顺庆区', '高坪区', '嘉陵区', '阆中市', '南部县', '营山县', '蓬安县', '仪陇县', '西充县']
def extract_district(region):
    for district in nanchong_districts:
        if district[:2] in region:
            return district
    return '其他'
df['区域'] = df['所属区域'].apply(extract_district)
df['区域编码'] = le.fit_transform(df['区域'])
'''
# 区域和区域编码的对应关系
      区域  区域编码
0    高坪区     8
1    顺庆区     7
6    嘉陵区     2
8    阆中市     6
26   营山县     3
30   南部县     1
40   蓬安县     4
487  西充县     5
739  仪陇县     0
'''

# 房龄 & 房龄编码
house_year = ['2年内','2-5年',]
def extract_house_year(year):
    for district in house_year:
        if district in year:
            return district
    return '其他'
df['房龄'] = df['房龄'].apply(extract_house_year)
df['房龄编码'] = le.fit_transform(df['房龄'])
'''
# 房龄和房龄编码的对应关系
        房龄  房龄编码
0      2年内     1
1086  2-5年     0
'''

# 方位 & 方位编码
df['方位编码'] = le.fit_transform(df['方位'])
'''
# 查看方位和方位编码的对应关系
     方位  方位编码
0     南     5
1    南北     6
8    东西     3
34   西南     9
38   东南     2
41   西北     8
72    东     0
84   东北     1
96    北     4
101   西     7
'''

# 删除不需要的列
df = df.drop(['标题','方位','户型', '楼层', '时间', '所属小区', '所属区域', '楼层类别','房龄','区域'], axis=1)

# 调整df的列顺序
df = df[['面积','方位编码','区域编码','总价', '均价', '房间数', '客厅数', '卫生间数', '建造年份', '楼层编码', '总楼层','房龄编码']]
'''
# 处理后的数据格式
    面积  方位编码  区域编码     总价      均价    房间数  客厅数  卫生间数  建造年份  楼层编码   总楼层  房龄编码
0  81.97    5       8      260000.0   3172.0    3       2      2       2023     2       26.0      1
'''

# 3. 探索性数据分析
