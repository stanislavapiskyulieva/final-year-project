import os
import xml.etree.ElementTree as ET

data_dir = "../data/raw_harvard_tlink"
for file in os.listdir(data_dir):
    if not file.endswith('.xml'):
        continue

    # print(file)
    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    raw_replaced = raw.replace("&", "&amp;")
    # # parser = ET.XMLParser(encoding="utf-8")
    try:
        root = ET.fromstring(raw_replaced)
    except ET.ParseError as e:
        print("CANT OPEN FILE: {}".format(file))
        print(e)
        pass
    for event in root.findall("./TAGS/EVENT/[@type='TREATMENT']"):
        print(event)
