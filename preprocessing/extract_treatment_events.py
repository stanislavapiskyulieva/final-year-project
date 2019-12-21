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
    event_root = ET.Element("root")
    tlink_root = ET.Element("root")
    sectime_root = ET.Element("root")
    try:
        root = ET.fromstring(raw_replaced)
    except ET.ParseError as e:
        print("CANT OPEN FILE: {}".format(file))
        print(e)
        pass

    event_ids = set()
    event_idsStartInd = set()
    for event in root.findall("./TAGS/EVENT/[@type='TREATMENT']"):
        if event.attrib['start'] in event_idsStartInd:
            continue
        event_idsStartInd.add(event.attrib['start'])
        event_ids.add(event.attrib['id'])
        event_root.append(event)

    for eventID in event_ids:
        tlinks = root.findall("./TAGS/TLINK")
        for tlink in tlinks:
            tlinkAttr = tlink.attrib['id'].lower()
            if  not "sectime" in tlinkAttr:
                continue
            if tlink.attrib['fromID'] == eventID or tlink.attrib['toID'] == eventID:
                tlink_root.append(tlink)

    for secTime in root.findall("./TAGS/SECTIME"):
        sectime_root.append(secTime)

    if sectime_root.find("SECTIME") is None:
        sectimeAdmEl = ET.SubElement(sectime_root, "SECTIME")
        sectimeAdmEl.set("type", "ADMISSION")
        admissionTLINK =  root.find("./TAGS/TLINK/[@fromText='ADMISSION']")
        if admissionTLINK is None:
            admissionTLINK = root.find("./TAGS/TLINK/[@toText='ADMISSION']")
            admissionValue = admissionTLINK.attrib['fromText']
        else:
            admissionValue = admissionTLINK.attrib['toText']

        sectimeAdmEl.set("text", admissionValue)
        sectimeDisEl = ET.SubElement(sectime_root, "SECTIME")
        sectimeDisEl.set("type", "DISCHARGE")
        dischargeTLINK =  root.find("./TAGS/TLINK/[@fromText='DISCHARGE']")
        if dischargeTLINK is None:
            dischargeTLINK = root.find("./TAGS/TLINK/[@toText='DISCHARGE']")
            dischargeValue = dischargeTLINK.attrib['fromText']
        else:
            dischargeValue = dischargeTLINK.attrib['toText']
        sectimeDisEl.set("text", dischargeValue)

    event_tree = ET.ElementTree(event_root)
    tlink_tree = ET.ElementTree(tlink_root)
    sectime_tree = ET.ElementTree(sectime_root)

    event_tree.write(data_dir + "/treatment_events/" + os.path.splitext(file)[0] + ".event.xml")
    tlink_tree.write(data_dir + "/treatment_events/" + os.path.splitext(file)[0] + ".tlink.xml")
    sectime_tree.write(data_dir + "/treatment_events/" + os.path.splitext(file)[0] + ".sectime.xml")
