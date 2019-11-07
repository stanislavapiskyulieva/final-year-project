import os
import xml.etree.ElementTree as ET
import copy

data_dir = "../data/raw_harvard_tlink"
for file in os.listdir(data_dir):
    if not file.endswith('.xml'):
        continue
    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    raw_replaced = raw.replace("&", "&amp;")
    new_root = ET.Element("root")
    try:
        root = ET.fromstring(raw_replaced)
    except ET.ParseError as e:
        print("CANT OPEN FILE: {}".format(file))
        print(e)
        pass

    event_ids = set()
    for event in root.findall("./TAGS/EVENT/[@type='TREATMENT']"):
        event_ids.add(event.attrib['id'])
        new_root.append(event)

    for eventID in event_ids:
        tlinks = root.findall("./TAGS/TLINK")
        for tlink in tlinks:
            if  not "SECTIME" in tlink.attrib['id']:
                continue
            if tlink.attrib['fromID'] == eventID or tlink.attrib['toID'] == eventID:
                new_root.append(tlink)

    treatment_tree = ET.ElementTree(new_root)
    treatment_tree.write(data_dir + "/treatment_events/" + file)
