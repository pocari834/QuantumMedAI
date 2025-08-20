import xml.etree.ElementTree as ET


def extract_text_from_xml(xml_path):
    """从OpenI的XML文件中提取诊断文本"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 示例：提取<AbstractText>标签内容（根据实际XML结构调整）
    findings = []
    for elem in root.iter('AbstractText'):
        if elem.text and elem.attrib.get('Label') == 'FINDINGS':
            findings.append(elem.text.strip())

    return " ".join(findings) if findings else ""