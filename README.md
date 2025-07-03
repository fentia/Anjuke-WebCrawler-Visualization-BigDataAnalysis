# Anjuke-WebCrawler-Visualization-BigDataAnalysis

本项目为安居客二手房价数据的全流程实战项目，涵盖数据采集、清洗、统计分析及可视化，适用于成都/南充等地二手房市场研究。项目采用 Selenium 爬虫自动采集数据，结合 pandas、numpy、scikit-learn 等主流数据分析库，支持多维度统计与可视化展示。

## 目录结构

```
case/                   # 案例与建模代码（如回归、分类等）
chromedriver/           # Chrome驱动及下载链接
project/
  ├── 南充二手房价爬虫.py       # 主爬虫程序
  ├── data/
  │   ├── CleanData/           # 清洗后数据（json/xlsx）
  │   └── OriginalData/        # 原始爬取数据
  ├── dataProcessing/          # 数据清洗与统计分析脚本
  ├── dataAnalysis/            # 深度数据分析与报告
  │   └── result/              # 分析结果输出
  └── dataVisualization/       # 数据可视化前端页面（HTML+JS+CSS）
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
- 主要依赖库：`pandas`、`numpy`、`scikit-learn`、`selenium`、`matplotlib` 等

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

## 联系方式

如有问题欢迎提 issue 或联系作者。