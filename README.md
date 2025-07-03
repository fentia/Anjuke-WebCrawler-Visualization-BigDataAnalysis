# Anjuke-WebCrawler-Visualization-BigDataAnalysis

本项目为安居客二手房价数据的全流程实战项目，涵盖数据采集、清洗、统计分析及可视化，适用于成都/南充等地二手房市场研究。项目采用 Selenium 爬虫自动采集数据，结合 pandas、numpy、scikit-learn 等主流数据分析库，支持多维度统计与可视化展示。

## 在线演示

你可以通过在线网页访问项目可视化与分析结果：[https://ajkksh.pages.dev/](https://ajkksh.pages.dev/)

## 目录结构

```
case/                   # 案例与建模代码（如回归、分类等）
chromedriver/           # Chrome驱动及下载链接
project/
  ├── 南充二手房价爬虫.py       # 主爬虫程序
  ├── data/
  │   ├── CleanData/           # 清洗后数据（json/xlsx）
  │   └── OriginalData/        # 爬取到的原始数据
  ├── dataProcessing/          # 数据清洗与数据统计脚本
  ├── dataAnalysis/            # 深度数据分析
  │   └── result/              # 数据分析结果
  └── dataVisualization/       # 数据可视化前端页面（HTML+JS+CSS）
      ├── assets/              # 静态资源
      ├── js/                  # 前端交互与可视化脚本
      ├── css/                 # 样式表文件
      ├── data/                # 前端展示用的静态数据
      └── index.html           # 可视化主页面入
```

## 项目功能

- 自动化爬取安居客成都/南充二手房数据（Selenium）
- 数据清洗与格式标准化（支持多种格式转换）
- 多维度统计分析（均价、面积、户型、建造年份、区域等）
- 数据分析报告生成与结果输出
- 交互式数据可视化（前端 HTML+JS）

## 环境依赖

- Python 3.8+
- Chrome 浏览器及对应版本 chromedriver
- 主要依赖库：`pandas`、`numpy`、`scikit-learn`、`selenium`、`matplotlib`、`seaborn`、`joblib` 等

### 推荐安装二进制预编译库

部分依赖（如 sklearn、scipy 等）建议使用二进制包安装，避免本地编译问题：

```sh
pip install --only-binary :all: scipy
pip install --only-binary=:all: scikit-learn
pip install --only-binary=:all: statsmodels
```

## 使用说明

1. 配置 Chrome 浏览器及 chromedriver 路径。
2. 运行爬虫脚本采集数据：
   ```sh
   python project/南充二手房价爬虫.py
   ```
3. 使用 `project/dataProcessing/` 下脚本进行数据清洗与统计分析。
4. 运行 `project/dataAnalysis/` 下脚本生成分析报告。
5. 打开 `project/dataVisualization/index.html` 查看可视化结果，可根据需要调整参数。

## 数据分析与建模说明

### 数值编码说明

在 [`project/dataAnalysis/dataAnalysis.py`](project/dataAnalysis/dataAnalysis.py) 的数据预处理过程中，部分分类特征被编码为数值，便于后续建模：

- **方位编码**：如东=0，东北=1，东南=2，东西=3，北=4，南=5，西=7，西北=8，西南=9，未知=6。
- **区域编码**：顺庆区=0，高坪区=1，嘉陵区=2，阆中市=3，南部县=4，营山县=5，蓬安县=6，仪陇县=7，西充县=8。
- **楼层编码**：高层=2，中层=0，低层=1，未知=3。
- **房龄编码**：2年内=1，2-5年=0。

具体编码可参考 [`data_preprocessing`](project/dataAnalysis/dataAnalysis.py) 函数中的注释和实现。

### 主要特征字段

建模时主要使用以下特征：

- 面积、方位编码、区域编码、房间数、客厅数、卫生间数、建造年份、楼层编码、总楼层、房龄编码

### 模型与指标

采用随机森林回归模型（`RandomForestRegressor`），并对特征进行标准化。模型评估指标包括：

- **MSE（均方误差）**：衡量预测值与真实值的平均平方差
- **MAE（平均绝对误差）**：预测值与真实值的平均绝对差
- **R2（决定系数）**：反映模型对数据的拟合优度，越接近1越好

模型训练与评估代码详见 [`data_analysis_modeling`](project/dataAnalysis/dataAnalysis.py) 函数，示例输出：

```
MSE: 241601125406.43768
MAE: 98201.04906191427
R2: 0.8741610260856343
```

## 联系方式

如有问题欢迎提 issue 或联系作者。