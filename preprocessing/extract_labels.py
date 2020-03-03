import os
import xml.etree.ElementTree as ET
import copy

def getLabels(file, drugEvents, data_dir):
    data_dir = data_dir + "/treatment_events"
    fileName = file.split('.')[0] + ".event.xml"
    f = open(os.path.join(data_dir, fileName), 'r')
    rawEvent = f.read()
    try:
        event_root = ET.fromstring(rawEvent)
    except ET.ParseError as e:
        print("CANT OPEN FILE: {}".format(fileName))
        print(e)
        pass

    f = open(os.path.join(data_dir, fileName.split('.')[0] + ".sectime.xml"), 'r')
    rawSecTime = f.read()
    secTime_root = ET.fromstring(rawSecTime)

    f = open(os.path.join(data_dir, fileName.split('.')[0] + ".tlink.xml"), 'r')
    rawTLINK = f.read()
    tlink_root = ET.fromstring(rawTLINK)

    labels = []
    admissionDate = 0
    dischargeDate = 0

    for section in secTime_root.findall("./SECTIME"):
        if section.attrib['type'].lower() == "admission":
            admissionDate = section.attrib['text']
        elif section.attrib['type'].lower() == "discharge":
            dischargeDate = section.attrib['text']

    sorted_event_root = sorted(event_root, key=lambda child: (child.tag,int(child.get('start'))))

    for event in sorted_event_root:
        if(event.attrib['text'] not in drugEvents):
            continue
        tlink = tlink_root.find('./TLINK[@fromID="{value}"]'.format(value=event.attrib['id']))
        if tlink is None:
            labels.append("during")
            continue
        tlinkType =  tlink.attrib['type'].lower()
        tlinkValue = tlink.attrib['toText'].lower()
        if tlinkType == "overlap" or tlinkType == "before_overlap":
            labels.append("during")
        elif tlinkValue == admissionDate or tlinkValue == "admission":
            if tlinkType == "before":
                labels.append("before")
            elif tlinkType == "after":
                labels.append("during")
        elif tlinkValue == dischargeDate or tlinkValue == "discharge":
            if tlinkType == "before":
                labels.append("during")
            elif tlinkType == "after":
                labels.append("after")
        else:
            labels.append("during")

    return labels
