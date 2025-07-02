# Anjuke-WebCrawler-Visualization-BigDataAnalysis

本项目为安居客二手房价数据爬虫、数据清洗、统计分析与可视化的综合实战，涵盖数据采集、处理、分析与可视化全流程。

## 目录结构

```
case/                   # 相关案例与建模代码
chromedriver/           # Chrome驱动及下载链接
project/
  ├── 南充二手房价爬虫.py         # 南充房价爬虫主程序
  ├── data/                    # 数据存储目录
  │   ├── CleanData/           # 清洗后数据
  │   └── OriginalData/        # 原始数据
  ├── dataProcessing/          # 数据处理与统计分析脚本
  └── dataVisualization/       # 数据可视化前端页面
```

## 主要功能

- 安居客成都/南充二手房价数据爬取（Selenium）
- 数据清洗与格式转换
- 多种统计分析（均价、面积、户型、建造年份等）
- 数据可视化（HTML+JS）

## 环境依赖

- Python 3.8+
- Chrome浏览器及对应版本的chromedriver
- 主要依赖库：`pandas`、`numpy`、`scikit-learn`、`selenium`、`matplotlib` 等

### 安装二进制预编译库

对于案例中的`ipynb`文件所涉及到的`sklearn`等需要编译的模块推荐使用二进制预编译库，避免本地编译问题。例如：

#### 安装 scipy
```
pip install --only-binary :all: scipy
```
#### 安装 scikit-learn
```
pip install --only-binary=:all: scikit-learn
```

## 使用说明

1. 配置好 Chrome 浏览器及 chromedriver 路径。
2. 运行爬虫脚本采集数据：
   ```
   python project/南充二手房价爬虫.py
   ```
3. 使用 `project/dataProcessing/` 下的脚本进行数据清洗与统计分析。
4. 自行配置 `project/dataVisualization/index.html` 中的参数，运行查看可视化结果。

## 联系方式

如有问题欢迎提 issue 或联系作者