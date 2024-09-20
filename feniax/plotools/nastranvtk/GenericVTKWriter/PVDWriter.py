import sys, os
import xml.etree.ElementTree as ET
import feniax.plotools.nastranvtk.FSUtils as FSUtils


def WritePVD(pvdfilename, filedata):
    """Write a Paraview Data file (vtk collection). This is used for
    time varying data
    Inputs:
    pvdfilename: ouput filename
    filedata: [[fname1,time],[fname2,time],...]"""

    xpvdfile = ET.Element(
        "VTKFile",
        {"type": "Collection", "version": "1.0", "byte_order": "LittleEndian"},
    )

    xcoldataset = ET.Element("Collection")

    xpvdfile.append(xcoldataset)

    for datalist in filedata:
        # Make sure this is a relative path - to allow copying of the pvd file
        fname = os.path.relpath(datalist[0], os.path.dirname(pvdfilename))
        time = datalist[1]
        xdataset = ET.Element(
            "DataSet", {"timestep": str(time), "part": str(0), "file": fname}
        )
        xcoldataset.append(xdataset)

    outFile = open(pvdfilename, "w")
    filestr = FSUtils.prettyPrint(xpvdfile)
    outFile.write(filestr)
    outFile.close()
