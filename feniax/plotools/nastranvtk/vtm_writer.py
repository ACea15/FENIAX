"""Code to write the top level vtm file that references all the vtu files (blocks)"""

import xml.etree.ElementTree as ET
import os
import feniax.plotools.nastranvtk.common as common


def writeVTM(filepath, femName, fileswritten, elemNames):
    xvtmfile = ET.Element(
        "VTKFile",
        {
            "type": "vtkMultiBlockDataSet",
            "version": "1.0",
            "byte_order": "LittleEndian",
        },
    )

    xmbdataset = ET.Element("vtkMultiBlockDataSet")

    xvtmfile.append(xmbdataset)

    for i, elem in enumerate(elemNames):
        xdataset = ET.Element(
            "DataSet", {"name": elem, "index": str(i), "file": fileswritten[i]}
        )
        xmbdataset.append(xdataset)

    filename = os.path.join(filepath, femName + ".vtm")
    outFile = open(filename, "w")
    filestr = common.prettyPrint(xvtmfile)
    outFile.write(str(filestr))
    outFile.close()
