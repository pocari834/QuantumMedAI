import os
import pandas as pd
from xml_parser import extract_text_from_xml


# 生成metadata.csv时添加标签提取逻辑
def generate_metadata():
    image_dir = "data/images"
    report_dir = "data/reports"

    records = []
    for xml_file in os.listdir(report_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(report_dir, xml_file)
            image_path = os.path.join(image_dir, xml_file.replace('.xml', '.png'))

            if os.path.exists(image_path):
                # 从XML中提取标签（示例：假设FINDINGS包含关键词'pneumonia'）
                text = extract_text_from_xml(xml_path)
                label = 1 if 'pneumonia' in text.lower() else 0
                records.append({
                    'image_path': image_path,
                    'report_path': xml_path,
                    'label': label,
                    'findings_text': text  # 可选：保存原始文本
                })

    pd.DataFrame(records).to_csv("data/metadata.csv", index=False)


if __name__ == "__main__":
    generate_metadata()