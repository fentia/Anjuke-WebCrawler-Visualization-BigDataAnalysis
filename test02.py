import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# 打开一个新的命令提示符窗口
subprocess.Popen('cmd', shell=True)
# 在新的命令提示符窗口执行命令打开谷歌浏览器
subprocess.Popen('"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222', shell=True)

# 用selenium连接到打开的谷歌浏览器窗口
options = Options()
options.add_experimental_option("debuggerAddress", "localhost:9222")
Google_driver_path = Service(executable_path=r'chromedriver-win64\chromedriver.exe')
driver = webdriver.Chrome(service=Google_driver_path,options=options)