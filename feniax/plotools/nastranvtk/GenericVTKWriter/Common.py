"""Common.py a place for miscellaneous useful functions"""

import numpy, struct
import os, errno
import xml.etree.ElementTree as ET


def MakeDir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def prettyPrint(element):
    indent(element)
    txt = ET.tostring(element)
    return txt


def bincalc(binar):
    """Convert a string to a binary base64 representation"""
    binar = binar.encode("base64").strip().replace("\n", "")
    size = len(binar) * 4
    size = struct.pack("<i", size).encode("base64").strip()
    binar = size + binar

    return binar
