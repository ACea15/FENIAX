import os, errno
try:
    import xml.etree.ElementTree as ET
except:
    raise ImportError("** ERROR: Failed to import <xml.etree.ElementTree> module. Are you running Python > v2.5?")

def MakeDir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def prettyPrint(element):
    indent(element)
    txt = ET.tostring(element)
    return txt
