import sys, os
import xml.etree.ElementTree as ET

import Common


def WriteVTM(vtmfilename, filedata):
    """Write a VTK multiblock dataset file
    Inputs:
    vtmfilename: output filename
    filedata: [[fname1,dataname1],[fname2,dataname2],...]"""

    # print fileswritten
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

    for i, datalist in enumerate(filedata):
        fname = os.path.relpath(datalist[0], os.path.dirname(vtmfilename))
        dataname = datalist[1]

        xdataset = ET.Element(
            "DataSet", {"name": dataname, "index": str(i), "file": fname}
        )
        xmbdataset.append(xdataset)

    outFile = open(vtmfilename, "w")
    filestr = Common.prettyPrint(xvtmfile)
    outFile.write(filestr)
    outFile.close()
