# PracticalTraining
爬虫 & 可视乎 & 大数据分析实战

### 一、安装二进制预编译库
Python pip安装地方放库的时候，经常会需要编译安装，而很多时候环境导致编译失败，不如直接安装二进制预编译库来的方便快捷，只要在pip install的时候，带上参数：```--only-binary :all:``` 即可，例如：
#### 安装 scipy
```
pip install --only-binary :all: scipy
```
#### 安装sklearn  
```
pip install --only-binary=:all: scikit-learn
```
这样就直接安装合适的预编译库，无需现场编译了
