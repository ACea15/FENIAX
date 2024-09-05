# ==========================================================================
# Project: FSUtils - EADS Innovation Works - Copyright (c) 2012, All rights reserved.
# Type   : FlowSimulator module python script
# File   : FSUtils.py
# Author : John Pattinson
# ==========================================================================
# DISCLAIMER
# This tool is provided on the strict understanding that the
# user accepts responsibility for checking and validating the outputs.
# Although every effort has been made to ensure the tools function
# correctly and consistently, no liability is accepted for any errors
# discovered.
# ==========================================================================
# DESCRIPTION
# FSUtils -- Miscellaneous utils
# ==========================================================================

import os, errno, re, struct

try:
    from collections import OrderedDict
except ImportError:
    from myOrderedDict import OrderedDict

import xml.etree.ElementTree as ET


def MakeDir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass  # If dirname exists ignore
        else:
            raise


def FSMakeDir(clac, dirname):
    if clac.GetProcID() == 0:
        MakeDir(dirname)


def str2bool(boolstr):
    if boolstr.lower() in ["false", "0"]:
        return False
    elif boolstr.lower() in ["true", "1"]:
        return True
    else:
        raise ValueError(
            "Dont understand boolstr value: %s, use true or false" % boolstr
        )


def FSPrint(*args, **kwargs):
    """A simple function that calls print with spaces between all the arguments. If there is a parallel communication context (clac) it only prints on the first process"""

    thestr = "".join([str(s) + " " for s in args])
    if kwargs.has_key("clac"):
        clac = kwargs["clac"]
        if clac is not None:
            if clac.GetProcID() == 0:
                print(thestr)
    else:
        print(thestr)


def Find(xml, findstr):
    """Python 2.6.6 support -- elementtree in python2.6 does not support xpath properly"""
    # Drop in replacement for xml.find(.//tag[@attrname="val"])

    # Only attribute finding does not work so can use this functionality for some of this.

    # Start at top -- split find string by separators
    # print
    alltags = findstr.split("/")
    tags = []
    for tag in alltags:
        if tag not in ["", "."]:
            tags.append(tag)
    # print "TAGS:",tags
    if tags[0] not in ["", "."]:
        tagstr = tags[0]
        findattrs = re.findall(r"(\[.*?\])", tagstr)
        if len(findattrs) > 0:
            # print "ATTRS FOUND",tagstr
            elemlist = []
            if len(findattrs) == 1:
                tagstr = tagstr.replace(findattrs[0], "")
                # print "",tagstr
                elems = xml.findall(".//" + tagstr)
                attr = findattrs[0].split("=")[0].replace("@", "").replace("[", "")
                value = findattrs[0].split("=")[1].replace("'", "").replace("]", "")
                for elem in elems:
                    if elem.attrib.has_key(attr):
                        if elem.attrib[attr] == value:
                            # print "ELEM_FOUND"
                            elemlist.append(elem)
            else:
                raise ValueError("tag not properly formed %s " % tag)
        else:
            # print "NO ATTR SEARCH",tagstr
            elemlist = xml.findall(".//" + tagstr)

        # print "LEN ELEMSLIST",len(elemlist)
        if len(elemlist) > 0:
            newfindstr = ""
            for jtag in range(1, len(tags)):
                if jtag != len(tags) - 1:
                    newfindstr += tags[jtag] + "/"
                else:
                    newfindstr += tags[jtag]
            if newfindstr != "":
                foundelem = Find(elemlist[0], newfindstr)
            else:
                foundelem = elemlist[0]
        else:
            foundelem = None

    return foundelem


def xmlfind(xml, findstr):
    """Helper fuction for elementree in the presence of old versions of python"""

    if os.environ["RELEASE_TAG"] == "r_2_2_3":
        return Find(xml, findstr)
    else:
        return xml.find(findstr)


def PrintVersionInfo(filename, bzr_version, fsVars=None):
    versionStr = "=========== %25s version information ===========\n" % (
        os.path.basename(filename)
    )
    for key, val in bzr_version.version_info.items():
        if not key in ["clean"]:
            versionStr += "%13s : %s\n" % (key, val)
    versionStr += (
        "======================================================================\n"
    )

    if fsVars is not None:
        if fsVars.clac.GetProcID() == 0:
            print(versionStr)
    else:
        print(versionStr)


def PrintDict(inputdict, spaces=0):
    dictitems = inputdict.items()
    # Sort alphabetically by key name
    dictitems = sorted(dictitems, key=lambda x: x[0])

    dictstr = ""
    for key, value in dictitems:
        if isinstance(value, dict) or isinstance(value, OrderedDict):
            dictstr = dictstr + " " * spaces + "%25s: ------------------\n" % key
            dictstr = (
                dictstr
                + PrintDict(value, spaces + 5)
                + " " * spaces
                + "%25s  ------------------\n" % ""
            )
        else:
            dictstr = dictstr + " " * spaces + "%25s: %s\n" % (key, str(value))

    return dictstr


def PrintInputDict(input_data, fsVars=None):
    dictstr = (
        "===========================RUN INPUTS==================================\n"
    )

    dictstr = dictstr + PrintDict(input_data)

    dictstr = (
        dictstr
        + "=======================================================================\n"
    )

    if fsVars is not None:
        if fsVars.clac.GetProcID() == 0:
            print(dictstr)
    else:
        print(dictstr)


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


def bincalc(binar, size=4):
    """Convert a string to a binary base64 representation"""
    binar = binar.encode("base64").strip().replace("\n", "")
    strsize = len(binar) * size
    binsize = struct.pack("<i", strsize).encode("base64").strip()
    binar = binsize + binar

    return binar


def printmat(mat, name=None, fmt="%12.5g", tol=None):
    s = mat.shape
    if len(s) == 1:
        mat = mat.reshape((s[0], 1)).copy()

    ni, nj = mat.shape

    matstr = ""
    if name is not None:
        matstr = "%s\n" % name

    for i in range(ni):
        for j in range(nj):
            num = mat[i, j]
            if tol is not None:
                if abs(mat[i, j]) < tol:
                    num = 0
            matstr += fmt % num
        matstr += "\n"
    return matstr


def debugPath(xml, xpath):
    """A little function to make it possible to work out where the xpath has gone wrong"""
    if ".//" in xpath:
        xpath = xpath.replace(".//", "")
    else:
        # Warn?
        pass

    xpathbits = xpath.split("/")

    testxpath = "./"
    for xpathbit in xpathbits:
        testxpath += "/" + xpathbit
        if xml.find(testxpath) is None:
            print("NotFound:", testxpath)
            break
        else:
            print("Found   :", testxpath)


def buildNumRangeFromString(fullRangeStr):
    """Build a list of ints from a string like 1-10,11:15,2 2-6"""
    rangeStrs = fullRangeStr.lstrip().rstrip().replace(" ", ",").split(",")
    numRangeInt = []
    for rangeStr in rangeStrs:
        if len(rangeStr) > 0:
            if rangeStr.find("-") >= 0 or rangeStr.find(":") >= 0:
                rangeStr = rangeStr.replace("-", ",").replace(":", ",")
                rangeList = eval("range(" + rangeStr + ")")
                rangeList.append(rangeList[-1] + 1)
                numRangeInt.extend(rangeList)
            else:
                numRangeInt.append(int(rangeStr))

    # sort and remove duplicates
    numranges_sorted = sorted(list(set(numRangeInt)))
    return numranges_sorted
