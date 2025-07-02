#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#数据预处理" data-toc-modified-id="数据预处理-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据预处理</a></span><ul class="toc-item"><li><span><a href="#数据导入" data-toc-modified-id="数据导入-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>数据导入</a></span></li><li><span><a href="#原始数据基本信息" data-toc-modified-id="原始数据基本信息-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>原始数据基本信息</a></span></li><li><span><a href="#缺失值-重复行-数据清洗" data-toc-modified-id="缺失值-重复行-数据清洗-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>缺失值 重复行 数据清洗</a></span></li></ul></li><li><span><a href="#探索性分析-(EDA)" data-toc-modified-id="探索性分析-(EDA)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>探索性分析 (EDA)</a></span><ul class="toc-item"><li><span><a href="#基本统计量" data-toc-modified-id="基本统计量-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>基本统计量</a></span></li><li><span><a href="#近五年成都市各区域二手房均价" data-toc-modified-id="近五年成都市各区域二手房均价-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>近五年成都市各区域二手房均价</a></span></li><li><span><a href="#近五年成都市各区域二手房均价箱线图" data-toc-modified-id="近五年成都市各区域二手房均价箱线图-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>近五年成都市各区域二手房均价箱线图</a></span></li><li><span><a href="#面积(㎡)带KDE的直方图及散点图" data-toc-modified-id="面积(㎡)带KDE的直方图及散点图-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>面积(㎡)带KDE的直方图及散点图</a></span></li><li><span><a href="#总价(万)带KDE的直方图与散点图" data-toc-modified-id="总价(万)带KDE的直方图与散点图-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>总价(万)带KDE的直方图与散点图</a></span></li><li><span><a href="#不同户型的频次分布" data-toc-modified-id="不同户型的频次分布-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>不同户型的频次分布</a></span></li><li><span><a href="#不同方位的频次分布" data-toc-modified-id="不同方位的频次分布-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>不同方位的频次分布</a></span></li><li><span><a href="#不同建造年份的二手房均价对比" data-toc-modified-id="不同建造年份的二手房均价对比-2.8"><span class="toc-item-num">2.8&nbsp;&nbsp;</span>不同建造年份的二手房均价对比</a></span></li></ul></li><li><span><a href="#回归模型(决策树_随机森林_XGBoost)" data-toc-modified-id="回归模型(决策树_随机森林_XGBoost)-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>回归模型(决策树_随机森林_XGBoost)</a></span><ul class="toc-item"><li><span><a href="#特征提取" data-toc-modified-id="特征提取-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>特征提取</a></span></li><li><span><a href="#相关系数热力图" data-toc-modified-id="相关系数热力图-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>相关系数热力图</a></span></li><li><span><a href="#模型训练与评估" data-toc-modified-id="模型训练与评估-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>模型训练与评估</a></span></li><li><span><a href="#模型性能结果可视化" data-toc-modified-id="模型性能结果可视化-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>模型性能结果可视化</a></span></li><li><span><a href="#绘制最优模型的拟合曲线" data-toc-modified-id="绘制最优模型的拟合曲线-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>绘制最优模型的拟合曲线</a></span></li><li><span><a href="#随机森林特征重要性得分图" data-toc-modified-id="随机森林特征重要性得分图-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>随机森林特征重要性得分图</a></span></li></ul></li></ul></div>

# In[ ]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import os
if not os.path.exists("安居客_成都二手房价数据分析中的图"):
    os.makedirs("安居客_成都二手房价数据分析中的图")


# # 数据预处理

# ## 数据导入 

# In[2]:


data = pd.read_excel('数据.xlsx')
data


# ## 原始数据基本信息

# In[3]:


data.info()


# ## 缺失值 重复行 数据清洗

# In[ ]:


# 1. 剔除缺失值和重复行
data.dropna(inplace=True)
duplicates = data.duplicated() # 检查是否有重复行
data.drop_duplicates(inplace=True)

# 2. 删除标题列，并将户型列的格式改为紧凑形式
data.drop(columns=['标题'], inplace=True)
data['户型'] = data['户型'].str.replace(' ', '')

# 3. 提取面积列的有效数据，并更改列名为面积(㎡)
data['面积(㎡)'] = data['面积'].str.replace('㎡', '').astype(float)
data.drop(columns=['面积'], inplace=True)

# 4. 剔除楼层列中类似“共1-5层”的数据
data = data[~data['楼层'].str.contains('共[1-5]层')]
#data = data[~data['楼层'].str.contains('地下(17层)')]
# 处理楼层列，删除括号及其中内容
data['楼层'] = data['楼层'].str.extract(r'([^\(]+)')[0]

# 5. 提取区域列中的有效信息
def extract_area(area_str): 
    valid_areas = ['武侯', '高新区', '成华', '金牛', '锦江', '郫都', '龙泉驿', '新都', '青羊', '双流', '温江', '天府新区', '都江堰', '崇州', '青白江', '金堂', '彭州', 
'邛崃市', '高新西区', '新津', '大邑', '蒲江', '成都周边', '简阳', '东部新区']
    for area in valid_areas:
        if area in area_str:
            return area
    return None

data['所属区域'] = data['所属区域'].apply(extract_area)
data.dropna(subset=['所属区域'], inplace=True)

# 6. 提取总价列中的有效信息，并更改列名为总价(万)
data['总价(万)'] = data['总价'].str.replace('万', '').astype(float)
data.drop(columns=['总价'], inplace=True)

# 7. 提取均价列中的有效信息，并更改列名为均价(元/㎡)
data['均价(元/㎡)'] = data['均价'].str.replace('元/㎡', '').astype(float)
data.drop(columns=['均价'], inplace=True)

# 8. 提取时间列有效信息
data['建造时间'] = data['时间'].str.extract(r'(\d{4})').astype(int)
data = data.drop(columns=['时间'])

#print(df.isnull().sum())# 检查缺失值情况
print("具有重复值的行数：", duplicates.sum()) # 输出具有重复值的行数
data


# In[5]:


data.info()


# # 探索性分析 (EDA)

# ## 基本统计量

# In[6]:


data.describe()


# In[7]:


# 输出每列所包含类别及相应数量
columns_of_interest = ['方位', '楼层', '建造时间', '所属区域', '房龄']

for column in columns_of_interest:
    print(f"Column: {column}")
    print(data[column].value_counts())
    print("\n")


# ## 近五年成都市各区域二手房均价

# In[ ]:


area_avg_price = data.groupby('所属区域')['均价(元/㎡)'].mean().reset_index() # 按区域计算均价的平均值，并按均价从高到低排序
area_avg_price = area_avg_price.sort_values(by='均价(元/㎡)', ascending=False)
sns.set(style="whitegrid") # 设置绘图风格
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
palette = sns.color_palette("Blues", len(area_avg_price))[::-1] # 自定义调色板
plt.figure(figsize=(14, 10))
bar_plot = sns.barplot(x='均价(元/㎡)', y='所属区域', data=area_avg_price, palette=palette)
plt.title('近五年成都市各区域二手房均价', fontsize=20, weight='bold', color='navy')
plt.xlabel('均价 (元/㎡)', fontsize=16, weight='bold', color='darkblue')
plt.ylabel('所属区域', fontsize=16, weight='bold', color='darkblue')
for index, value in enumerate(area_avg_price['均价(元/㎡)']): # 显示均价数值
    bar_plot.text(value + 100, index, f'{value:.2f}', color='black', ha="left", va="center", fontsize=12)
plt.xticks(fontsize=12, color='darkblue') # 调整刻度字体大小和颜色
plt.yticks(fontsize=12, color='darkblue')
plt.grid(axis='x', linestyle='--', alpha=0.7) # 添加网格线
sns.despine()
plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图\近五年成都市各区域二手房均价.png', dpi=300, facecolor='white')  # 保存图像为文件
plt.show()


# ## 近五年成都市各区域二手房均价箱线图

# In[ ]:


sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
palette = sns.color_palette("viridis", len(data['所属区域'].unique()))[::-1] # 自定义调色板
plt.figure(figsize=(14, 10)) # 绘制箱线图
box_plot = sns.boxplot(x='所属区域', y='均价(元/㎡)', data=data, palette=palette)
plt.title('近五年成都市各区域二手房均价箱线图', fontsize=20, weight='bold', color='navy')
plt.xlabel('所属区域', fontsize=16, weight='bold', color='darkblue')
plt.ylabel('均价 (元/㎡)', fontsize=16, weight='bold', color='darkblue')
plt.xticks(fontsize=12, color='darkblue', rotation=30)
plt.yticks(fontsize=12, color='darkblue')
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图\近五年成都市各区域二手房均价箱线图.png', dpi=300, facecolor='white')
plt.show()


# In[10]:


# 保存预处理后的数据
processed_file_path = '预处理后的数据.xlsx'
data.to_excel(processed_file_path, index=False)
print(f'预处理后的数据已保存到 {processed_file_path}')


# ## 面积(㎡)带KDE的直方图及散点图

# In[ ]:


from scipy.stats import gaussian_kde

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# 面积(㎡)的KDE图和直方图
x = data['面积(㎡)'].dropna()
kde = gaussian_kde(x)
x_range = np.linspace(x.min(), x.max(), 1000)
axes[0].hist(x, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
axes[0].plot(x_range, kde(x_range), color='blue', lw=2)
axes[0].set_title('面积(㎡)带KDE的条形图', fontsize=20, weight='bold', color='navy')
axes[0].set_xlabel('面积 (㎡)', fontsize=16, weight='bold', color='darkblue')
axes[0].set_ylabel('密度', fontsize=16, weight='bold', color='darkblue')
axes[0].tick_params(axis='x', colors='darkblue', labelsize=12)
axes[0].tick_params(axis='y', colors='darkblue', labelsize=12)

# 面积(㎡)与均价(元/㎡)的相关性散点图和回归直线
sns.regplot(x='面积(㎡)', y='均价(元/㎡)', data=data, ax=axes[1], scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'lw':2})
axes[1].set_title('面积(㎡)与均价(元/㎡)的散点图', fontsize=20, weight='bold', color='navy')
axes[1].set_xlabel('面积 (㎡)', fontsize=16, weight='bold', color='darkblue')
axes[1].set_ylabel('均价 (元/㎡)', fontsize=16, weight='bold', color='darkblue')
axes[1].tick_params(axis='x', colors='darkblue', labelsize=12)
axes[1].tick_params(axis='y', colors='darkblue', labelsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图\面积(㎡)带KDE的直方图与散点图.png', dpi=300, facecolor='white')
plt.show()


# ## 总价(万)带KDE的直方图与散点图

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# 总价(万)的KDE图和直方图
x = data['总价(万)'].dropna()
kde = gaussian_kde(x)
x_range = np.linspace(x.min(), x.max(), 1000)
axes[0].hist(x, bins=30, density=True, alpha=0.5, color='lightgreen', edgecolor='black')
axes[0].plot(x_range, kde(x_range), color='green', lw=2)
axes[0].set_title('总价(万)带KDE的条形图', fontsize=20, weight='bold', color='navy')
axes[0].set_xlabel('总价 (万)', fontsize=16, weight='bold', color='darkblue')
axes[0].set_ylabel('密度', fontsize=16, weight='bold', color='darkblue')
axes[0].tick_params(axis='x', colors='darkblue', labelsize=12)
axes[0].tick_params(axis='y', colors='darkblue', labelsize=12)

# 总价(万)与均价(元/㎡)的相关性散点图和回归直线
sns.regplot(x='总价(万)', y='均价(元/㎡)', data=data, ax=axes[1], scatter_kws={'alpha':0.6, 'color':'green'}, line_kws={'color':'red', 'lw':2})
axes[1].set_title('总价(万)与均价(元/㎡)的散点图', fontsize=20, weight='bold', color='navy')
axes[1].set_xlabel('总价 (万)', fontsize=16, weight='bold', color='darkblue')
axes[1].set_ylabel('均价 (元/㎡)', fontsize=16, weight='bold', color='darkblue')
axes[1].tick_params(axis='x', colors='darkblue', labelsize=12)
axes[1].tick_params(axis='y', colors='darkblue', labelsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图/总价(万)带KDE的直方图与散点图.png', dpi=300, facecolor='white')
plt.show()


# ## 不同户型的频次分布

# In[ ]:


house_type_counts = data['户型'].value_counts().reset_index()
house_type_counts.columns = ['户型', '频次']
top_5 = house_type_counts.iloc[:3]
others = pd.DataFrame([{'户型': '其它', '频次': house_type_counts.iloc[5:]['频次'].sum()}])
house_type_counts_modified = pd.concat([top_5, others], ignore_index=True)
plt.figure(figsize=(8, 8))
colors = sns.color_palette("YlGn", len(house_type_counts_modified))[::-1] # 自定义调色板
plt.pie(house_type_counts_modified['频次'], labels=house_type_counts_modified['户型'], autopct='%1.1f%%',
        startangle=90, colors=colors, textprops={'fontsize': 14}, wedgeprops={'edgecolor': 'w'})
plt.title('不同户型的频次分布', fontsize=20, weight='bold')
plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图\不同户型的频次分布.png', dpi=300, facecolor='white')
plt.show()


# ## 不同方位的频次分布

# In[ ]:


orientation_counts = data['方位'].value_counts().reset_index()
orientation_counts.columns = ['方位', '频次']
orientation_counts = orientation_counts.sort_values(by='频次', ascending=False)
palette = sns.color_palette("GnBu", len(area_avg_price))[::-1] # 自定义调色板
plt.figure(figsize=(14, 10))
bar_plot = sns.barplot(x='频次', y='方位', data=orientation_counts, palette=palette)
plt.title('不同方位的频次分布', fontsize=20, weight='bold', color='navy')
plt.xlabel('频次', fontsize=16, weight='bold', color='darkblue')
plt.ylabel('方位', fontsize=16, weight='bold', color='darkblue')
for index, value in enumerate(orientation_counts['频次']):
    bar_plot.text(value + 10, index, f'{value}', color='black', ha="left", va="center", fontsize=12)
plt.xticks(fontsize=12, color='darkblue')
plt.yticks(fontsize=12, color='darkblue')
plt.grid(axis='x', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图/不同方位的频次分布.png', dpi=300, facecolor='white')
plt.show()


# ## 不同建造年份的二手房均价对比

# In[ ]:


data['建造年份'] = data['建造时间']
data = data[(data['建造年份'] >= 2019) & (data['建造年份'] <= 2023)]
regions = ['武侯', '高新区', '成华', '金牛', '锦江']
data = data[data['所属区域'].str.contains('|'.join(regions))]
avg_price_by_year = data.groupby(['所属区域', '建造年份'])['均价(元/㎡)'].mean().unstack()

plt.figure(figsize=(14, 10))
palette = sns.color_palette("husl", len(regions))
for i, region in enumerate(regions):
    plt.plot(avg_price_by_year.columns.astype(int), avg_price_by_year.loc[region], marker='o', label=region, color=palette[i])

plt.title('2019-2023年成都市代表区域不同建造年份的二手房均价对比', fontsize=20, weight='bold', color='navy')
plt.xlabel('建造年份', fontsize=16, weight='bold', color='darkblue')
plt.ylabel('均价 (元/㎡)', fontsize=16, weight='bold', color='darkblue')
plt.xticks(ticks=avg_price_by_year.columns.astype(int), fontsize=12, color='darkblue')
plt.yticks(fontsize=12, color='darkblue')
plt.legend(title='所属区域', fontsize=12, title_fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('安居客_成都二手房价数据分析中的图/不同建造年份的二手房均价对比.png', dpi=300, facecolor='white')
plt.show()


# # 回归模型(决策树_随机森林_XGBoost)

# ## 特征提取

# In[16]:


file_path = '预处理后的数据.xlsx' # 读取Excel文件
data = pd.read_excel(file_path)
data


# In[17]:


# 剔除列名为 '房龄'、'所属小区' 的列
columns_to_drop = ['房龄', '所属小区']
data = data.drop(columns=columns_to_drop)
data.head()


# In[18]:


# 输出方位、所属区域、楼层 的种类
print(f"方位种类: {data['方位'].unique()}")
print(f"所属区域种类: {data['所属区域'].unique()}")
print(f"楼层种类: {data['楼层'].unique()}")


# In[19]:


# 基于户型列创建新特征
data[['室', '厅', '卫']] = data['户型'].str.extract(r'(\d)室(\d)厅(\d)卫').astype(int)
columns_to_drop = ['户型']
data = data.drop(columns=columns_to_drop)
data


# In[20]:


from sklearn.preprocessing import LabelEncoder

# 创建并应用标签编码器
def encode_column(data, column_name):
    le = LabelEncoder()
    data[column_name] = le.fit_transform(data[column_name])
    print(f"{column_name}编码映射:", dict(zip(le.classes_, le.transform(le.classes_))))
    return le

# 编码方位、楼层、所属区域列
le_orientation = encode_column(data, '方位')
le_floor = encode_column(data, '楼层')
le_area = encode_column(data, '所属区域')
data


# In[21]:


data.info()


# ## 相关系数热力图

# In[22]:


fig, ax = plt.subplots(figsize=(20, 12))
df_corr = data.corr()
cmap = sns.diverging_palette(240, 0, 90, 60, as_cmap=True)
heatmap =sns.heatmap(df_corr, annot=True, fmt=".3f", linewidths=5, cmap=cmap, square=True)
heatmap.set_title('Pearson相关系数热力图', fontsize=20)
for text in heatmap.get_yticklabels() + heatmap.get_xticklabels():
    text.set_fontsize(16)
for text in heatmap.texts:
    text.set_fontsize(16)
plt.xticks(fontsize=15, rotation=30)  # 设置x轴的刻度和标签
plt.yticks(fontsize=15, rotation=30)
plt.savefig('安居客_兰州二手房价数据分析中的图/Pearson相关系数热力图.png', dpi=300, facecolor='white')
plt.show()


# ## 模型训练与评估

# 四个常用的统计指标用于评估回归模型的性能，包括均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）和决定系数（\( R^2 \)）。这些指标的数学公式如下：
# 1. **均方误差 (MSE)**:
# $$
# \mathrm{MSE}=\frac{1}{n} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2
# $$
# 
# 其中， $y_i$ 是实际观测值， $\hat{y}_i$ 是预测值， $n$ 是样本数量。
# 
# 2. **均方根误差 (RMSE)**:
# $$
# \text { RMSE }=\sqrt{\frac{1}{n} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2}=\sqrt{\mathrm{MSE}}
# $$
# 
# RMSE 是 MSE 的平方根，提供了与原始数据相同单位的误差评估。
# 
# 3. **平均绝对误差 (MAE)**:
# $$
# \mathrm{MAE}=\frac{1}{n} \sum_{i=1}^n\left|y_i-\hat{y}_i\right|
# $$
# MAE 衡量预测值和实际值之间的平均绝对差异。
# 
# 4. **决定系数 $\left(R^2\right)$** :
# $$
# R^2=1-\frac{\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2}{\sum_{i=1}^n\left(y_i-\bar{y}\right)^2}
# $$
# 
# 其中， $\bar{y}$ 是 $y_i$ 的平均值。 $R^2$ 表示模型能够解释的数据变异性的比例。

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
X = data.drop(columns=['均价(元/㎡)','面积(㎡)']) # 特征和目标变量
y = data['均价(元/㎡)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30) # 划分训练集和测试集
scaler = StandardScaler() # 标准化处理
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 定义模型
models = {
    '决策树': DecisionTreeRegressor(random_state=42),
    '随机森林': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=35)
}
# 训练和评估模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        '均方误差(MSE)': mse,
        '均方根误差(RMSE)': rmse,
        '平均绝对误差(MAE)': mae,
        'R方': r2
    }
for name, metrics in results.items():
    print(f"{name}")
    print(f"均方误差(MSE): {metrics['均方误差(MSE)']:.4f}, 均方根误差(RMSE): {metrics['均方根误差(RMSE)']:.4f}, 平均绝对误差(MAE): {metrics['平均绝对误差(MAE)']:.4f}, R方: {metrics['R方']:.4f}\n")


# ## 模型性能结果可视化

# In[24]:


# 模型性能结果
performance_df = pd.DataFrame(results).T

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('模型性能比较', fontsize=20, fontweight='bold')
colors = sns.color_palette("pastel")

# 绘制每个子图并添加数值标签
def add_labels(ax, data, x, y):
    sns.barplot(ax=ax, x=x, y=y, data=data, palette=colors)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 10), textcoords = 'offset points',
                   fontsize=10, color='black')
    ax.set_title(y, fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

# 均方误差(MSE)
add_labels(axes[0, 0], performance_df.reset_index(), 'index', '均方误差(MSE)')
axes[0, 0].set_ylabel('误差值')

# 均方根误差(RMSE)
add_labels(axes[0, 1], performance_df.reset_index(), 'index', '均方根误差(RMSE)')
axes[0, 1].set_ylabel('误差值')

# 平均绝对误差(MAE)
add_labels(axes[1, 0], performance_df.reset_index(), 'index', '平均绝对误差(MAE)')
axes[1, 0].set_ylabel('误差值')

# R方
add_labels(axes[1, 1], performance_df.reset_index(), 'index', 'R方')
axes[1, 1].set_ylabel('得分')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('安居客_兰州二手房价数据分析中的图/模型性能比较(3个）.png', dpi=300, facecolor='white')
plt.show()


# ## 绘制最优模型的拟合曲线

# In[25]:


best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
plt.figure(figsize=(14, 6))

# 训练集拟合曲线
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.3)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.5) # 添加网格
plt.gca().spines['top'].set_linewidth(0)  # 去掉顶部边框
plt.gca().spines['right'].set_linewidth(0)  # 去掉右边框
plt.gca().spines['bottom'].set_linewidth(0.5)  # 设置底部边框粗细
plt.gca().spines['left'].set_linewidth(0.5)  # 设置左边框粗细
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('训练集拟合曲线')

# 测试集拟合曲线
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True, linestyle='--', alpha=0.5) # 添加网格
plt.gca().spines['top'].set_linewidth(0)  # 去掉顶部边框
plt.gca().spines['right'].set_linewidth(0)  # 去掉右边框
plt.gca().spines['bottom'].set_linewidth(0.5)  # 设置底部边框粗细
plt.gca().spines['left'].set_linewidth(0.5)  # 设置左边框粗细
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('测试集拟合曲线')

plt.suptitle('随机森林模型的拟合曲线', fontsize=16)
plt.savefig('安居客_兰州二手房价数据分析中的图/随机森林模型的拟合曲线.png', dpi=300, facecolor='white')
plt.show()


# ## 随机森林特征重要性得分图

# In[26]:


predictions = best_model.predict(X_train)
feature_importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(14, 10))
palette = sns.color_palette("Blues", len(feature_names))[::-1]  # 自定义调色板
sns.barplot(x='Importance', y='Feature', data=importance_df, palette=palette)
plt.title('随机森林模型的特征重要性', fontsize=20, color='darkblue')
plt.xlabel('得分', fontsize=15, color='darkblue')
plt.ylabel('特征', fontsize=15, color='darkblue')
plt.xticks(fontsize=12, color='darkblue')
plt.yticks(fontsize=12, color='darkblue')
plt.grid(axis='x', linestyle='--', alpha=0.7)
for index, value in enumerate(importance_df['Importance']):
    plt.text(value, index, f'{value:.4f}', color='black', ha="left", va="center", fontsize=12)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['top'].set_linewidth(0.5)
plt.gca().spines['left'].set_linewidth(0.5)
plt.tight_layout()
plt.savefig('安居客_兰州二手房价数据分析中的图/随机森林特征重要性得分图.png', dpi=300, facecolor='white')
plt.show()

