import os
import xml.etree.ElementTree as ET
import copy
import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

data_dir = "../data/raw_harvard_tlink/treatment_events"
for file in os.listdir(data_dir):
    if not file.endswith('.event.xml'):
        continue
    f = open(os.path.join(data_dir, file), 'r')
    rawEvent = f.read()
    try:
        event_root = ET.fromstring(rawEvent)
    except ET.ParseError as e:
        print("CANT OPEN FILE: {}".format(file))
        print(e)
        pass

    f = open(os.path.join(data_dir, file.split('.')[0] + ".sectime.xml"), 'r')
    rawSecTime = f.read()
    secTime_root = ET.fromstring(rawSecTime)

    f = open(os.path.join(data_dir, file.split('.')[0] + ".tlink.xml"), 'r')
    rawTLINK = f.read()
    tlink_root = ET.fromstring(rawTLINK)

    labels = []

    for section in secTime_root.findall("./SECTIME"):
        if section.attrib['type'].lower() == "admission":
            admissionDate = section.attrib['text']
        elif section.attrib['type'].lower() == "discharge":
            dischargeDate = section.attrib['text']

    sorted_event_root = sorted(event_root, key=lambda child: (child.tag,int(child.get('start'))))

    for event in sorted_event_root:
        print(event.attrib['start'])
        print(file)
        tlink = tlink_root.find('./TLINK[@fromID="{value}"]'.format(value=event.attrib['id']))
        if tlink is None:
            labels.append("n/a")
            continue
        tlinkType =  tlink.attrib['type'].lower()
        tlinkValue = tlink.attrib['toText']
        if tlinkType == "overlap" or tlinkType == "before_overlap":
            labels.append("during")
        elif tlinkValue == admissionDate:
            if tlinkType == "before":
                labels.append("before")
            elif tlinkType == "after":
                labels.append("during")
        elif tlinkValue == dischargeDate:
            if tlinkType == "before":
                labels.append("during")
            elif tlinkType == "after":
                labels.append("after")
    print(labels)
