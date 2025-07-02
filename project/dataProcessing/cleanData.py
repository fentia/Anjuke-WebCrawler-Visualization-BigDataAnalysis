'''
该脚本用于处理JSON文件，主要功能包括：
1. 查找并统计重复的键值对
2. 清洗数据，去除重复的键值对、换行符和空格
3. 将清洗后的数据保存到新的JSON文件和Excel文件中
'''

import re
import pandas as pd

def findDuplicateData(input_josn):
    '''
    该函数用于处理JSON文件，找到重复的键值对，并统计重复次数
    '''
    con_dict = {}
    with open(input_josn, 'r', encoding='utf-8') as infile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            # 提取 "键": "值" 对
            pairs = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', line)

            value = pairs[0][1]
            if value not in con_dict:
                con_dict[value] = 1
            else:
                con_dict[value] += 1

    con0 = 0
    for key, value in con_dict.items():
        if value > 1:
            print(f"'{key}'重复{value}次")
            con0 += 1
    print(f"重复数据共有{con0}条")

    con1 = 0
    for key, value in con_dict.items():
        if value == 1:
            con1 += 1
    print(f"未重复数据有{con1}条")

    print("预计去重后数据量为:", con1 + con0,'条')

def cleanData(input_json, output_josn, output_excel):
    '''
    该函数用于处理JSON文件，去除重复的键值对，去除换行符和空格
    并将清洗后的数据写入新的JSON文件和可选的Excel文件
    '''
    seen_titles = set()
    cleaned_data = []

    with open(input_json, 'r', encoding='utf-8') as infile, \
         open(output_josn, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            # 提取所有键值对
            pairs = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', line)

            if not pairs:
                continue

            # 拿出标题作为判断是否重复的依据
            title = pairs[0][1]
            if title in seen_titles:
                continue  # 重复标题，跳过
            seen_titles.add(title)

            cleaned_pairs = []
            record_dict = {}

            for key, value in pairs:
                if key == '标题':
                    cleaned_value = value
                else:
                    cleaned_value = value.replace('\\n', '') \
                                         .replace('\n', '') \
                                         .replace('\r', '') \
                                         .replace(' ', '') \
                                         .replace('\u3000', '')
                cleaned_pairs.append(f'"{key}": "{cleaned_value}"')
                record_dict[key] = cleaned_value

            # 写入清洗后的行
            cleaned_line = '{' + ', '.join(cleaned_pairs) + '},'
            outfile.write(cleaned_line + '\n')
            cleaned_data.append(record_dict)

    # 写入 Excel 文件
    if output_excel:
        df = pd.DataFrame(cleaned_data)
        df.to_excel(output_excel, index=False)

def main(input_json, output_josn, output_excel):
    '''
    主函数，调用去重和清洗函数
    '''
    findDuplicateData(input_json)  # 查找重复数据
    cleanData(input_json, output_josn, output_excel)  # 清洗数据并去重
    print(f"清洗后的数据已保存到 {output_josn} 和 {output_excel}")

# 使用示例
if __name__ == '__main__':
    input_json = 'project\data\OriginalData\data.json'
    output_josn = 'project\data\CleanData\cleanData.json'
    output_excel = 'project\data\CleanData\cleanData.xlsx'

    main(input_json, output_josn, output_excel)