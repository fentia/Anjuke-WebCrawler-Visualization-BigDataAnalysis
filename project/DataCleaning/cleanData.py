import re
import pandas as pd

def clean_data(input_josn,output_josn,output_excel):
    data_list = []

    with open(input_josn, 'r', encoding='utf-8') as infile, \
        open(output_josn, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            # 提取 "键": "值" 对
            pairs = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', line)

            cleaned_pairs = []
            kv_dict = {}  # 同时构建字典
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
                kv_dict[key] = cleaned_value

            cleaned_line = '{' + ', '.join(cleaned_pairs) + '},'
            outfile.write(cleaned_line + '\n')

            data_list.append(kv_dict)  # 添加字典对象到列表

    # 使用 pandas 写入 Excel
    df = pd.DataFrame(data_list)
    df.to_excel(output_excel, index=False)

if __name__ == '__main__':
    input_josn = 'project\data\OriginalData\data.json'
    output_josn = 'project\data\CleanData\data.json'
    output_excel = 'project\data\CleanData\data.xlsx'
    clean_data(input_josn,output_josn,output_excel)
    print(f"数据已清洗并保存到 {output_josn} 和 {output_excel}")