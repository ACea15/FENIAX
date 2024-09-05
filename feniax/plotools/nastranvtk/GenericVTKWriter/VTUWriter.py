import numpy as np
import os, sys, struct
import xml.etree.ElementTree as ET

import feniax.plotools.nastranvtk.FSUtils as FSUtils


def WriteVTU(
    vtufilename,
    points,
    connectivity,
    pointData={},
    cellData={},
    surface=True,
    datatype="string",
):
    """A generic .vtu (or vtk) writer
    Inputs:
    vtufilename  = The filename for the file
    points       = numpy array of size [numpoints x 3] containing coordinates
    connectivity = a list of size numcells containing sub lists with each cells connectivity
    pointdata    = a dictionary containing numpy arrays with point data the key is the dataset name
    celldata     = a dictionary containing numpy arrays with cell data
    surface      = Is this a surface or volume mesh
    """

    # check size of points here
    nnodes = points.shape[0]
    nelems = len(connectivity)

    # Root element
    xvtufile = ET.Element(
        "VTKFile",
        {"byte_order": "LittleEndian", "type": "UnstructuredGrid", "version": "0.1"},
    )
    xunstructgrid = ET.Element("UnstructuredGrid")
    xvtufile.append(xunstructgrid)

    xpiece = ET.Element("Piece", {"NumberOfCells": str(nelems)})
    xunstructgrid.append(xpiece)

    # Write points
    xpoints = ET.Element("Points")
    xpiece.append(xpoints)

    # We know how many points there are now so use that.
    xpiece.attrib["NumberOfPoints"] = str(nnodes)
    xpoint_chords = ET.Element(
        "DataArray", {"type": "Float64", "format": datatype, "NumberOfComponents": "3"}
    )
    if datatype == "binary":
        string = b""
    else:
        string = str()

    for inode in range(nnodes):
        nodepos = points[inode, :]
        if datatype == "binary":
            string = string + struct.pack("<ddd", nodepos[0], nodepos[1], nodepos[2])
        else:
            string = string + "".join(
                [repr(nodepos[0]), " ", repr(nodepos[1]), " ", repr(nodepos[2]), " "]
            )

    # if datatype=="binary":
    #    string=FSUtils.bincalc(string,size=8)

    xpoint_chords.text = string
    xpoints.append(xpoint_chords)

    # Point data
    xPointData = ET.Element("PointData")
    xpiece.append(xPointData)

    for var in sorted(pointData):
        # Check variable shape
        if pointData[var].shape[0] != nnodes:
            raise ValueError(
                "Point Data variable %s is the wrong length. Number of nodes: %i, length of variable: %i"
                % (var, nnodes, pointData[var].shape[0])
            )

        if len(pointData[var].shape) == 1:
            ncomponents = 1
            pointData[var] = np.reshape(pointData[var], (nnodes, 1))
        elif len(pointData[var].shape) == 2:
            ncomponents = pointData[var].shape[1]
        else:
            raise ValueError(
                "Point Data variable %s is not 1 or 2 dimensional. len(pointData[var].shape) = %i"
                % (var, len(pointData[var].shape))
            )

        if ncomponents not in [1, 3]:
            raise ValueError(
                "Not sure if number of compnents can be anything other than 1 or 3 (scalar or vector)."
            )

        # Sort out data type
        arrayDataType = pointData[var].dtype
        if np.issubdtype(arrayDataType, int):
            vtkDataType = "Int32"
            binspec = "<i"
            size = 4
        elif np.issubdtype(arrayDataType, float):
            vtkDataType = "Float64"
            binspec = "<d"
            size = 8
        else:
            raise RuntimeError(
                "Not sure what numpy.dtype: %s translates to in vtk"
                % (str(arrayDataType))
            )

        xpointvar = ET.Element(
            "DataArray",
            {
                "type": vtkDataType,
                "NumberOfComponents": str(ncomponents),
                "format": datatype,
                "Name": var,
            },
        )
        if datatype == "binary":
            string = b""
        else:
            string = str()
        for inode in range(nnodes):
            val = pointData[var][inode, :]
            for icomponent in range(ncomponents):
                if datatype == "binary":
                    string = string + struct.pack(binspec, val[icomponent])
                else:
                    string = string + "".join([repr(val[icomponent]), " "])

        # if datatype=="binary":
        #     string=FSUtils.bincalc(string,size)

        xpointvar.text = string
        xPointData.append(xpointvar)

    # Write elements
    xcells = ET.Element("Cells")
    xpiece.append(xcells)

    xcell_connectivity = ET.Element(
        "DataArray", {"type": "Int32", "format": datatype, "Name": "connectivity"}
    )

    # Cell Connectivity data
    # string = str()
    # offset_string = str()
    offset_pos = 0
    if datatype == "binary":
        string = b""
    else:
        string = str()
    if datatype == "binary":
        offset_string = b""
    else:
        offset_string = str()

    for i in range(len(connectivity)):
        offset_pos += len(connectivity[i])
        # print offset_pos
        if datatype == "binary":
            offset_string = offset_string + struct.pack("<i", offset_pos)
        else:
            offset_string = offset_string + "".join([repr(offset_pos), " "])
        for j in range((len(connectivity[i]))):
            if datatype == "binary":
                string = string + struct.pack("<i", connectivity[i][j])
            else:
                string = string + "".join([repr(int(connectivity[i][j])), " "])

    # if datatype=="binary":
    #     offset_string = FSUtils.bincalc(offset_string)
    #     string = FSUtils.bincalc(string)

    xcell_connectivity.text = string
    xcells.append(xcell_connectivity)

    xcell_offsets = ET.Element(
        "DataArray", {"type": "Int32", "format": datatype, "Name": "offsets"}
    )
    xcell_offsets.text = offset_string
    xcells.append(xcell_offsets)

    # Element Types
    xcell_types = ET.Element(
        "DataArray", {"type": "Int32", "format": datatype, "Name": "types"}
    )

    # Mapping to VTK cell types based on the number of nodes
    if surface:  # Surface mesh
        elementTypeMapping = {
            1: 1,  # Point
            2: 3,  # Line
            3: 5,  # Triangle
            4: 9,  # Quad
        }
    else:  # Volume mesh
        elementTypeMapping = {
            4: 10,  # Tetra
            8: 12,  # Hexa
            6: 13,  # Prism
            5: 13,  # Pyramid
        }

    # string = str()
    if datatype == "binary":
        string = b""
    else:
        string = str()

    for i in range(len(connectivity)):
        try:
            if datatype == "binary":
                string += struct.pack("<i", elementTypeMapping[len(connectivity[i])])
            else:
                string = string + repr(elementTypeMapping[len(connectivity[i])]) + " "
        except KeyError:
            raise KeyError(
                "Type for cells with %i nodes not defined yet" % len(connectivity[i])
            )

    # if datatype=="binary":
    #     string=FSUtils.bincalc(string,size=4)

    xcell_types.text = string

    xcells.append(xcell_types)

    # Cell Data

    xCellData = ET.Element("CellData")
    xpiece.append(xCellData)

    for var in sorted(cellData):
        # Check variable shape
        if cellData[var].shape[0] != nelems:
            raise ValueError(
                "Cell Data variable %s is the wrong length. Number of nodes: %i, length of variable: %i"
                % (var, nnodes, cellData[var].shape[0])
            )

        if len(cellData[var].shape) == 1:
            ncomponents = 1
            cellData[var] = np.reshape(cellData[var], (nelems, 1))
        elif len(cellData[var].shape) == 2:
            ncomponents = cellData[var].shape[1]
        else:
            raise ValueError(
                "Cell Data variable %s is not 1 or 2 dimensional. len(cellData[var].shape) = %i"
                % (var, len(cellData[var].shape))
            )

        # Sort out data type
        arrayDataType = cellData[var].dtype
        if np.issubdtype(arrayDataType, int):
            vtkDataType = "Int32"
            binspec = "<i"
        elif np.issubdtype(arrayDataType, float):
            vtkDataType = "Float64"
            binspec = "<d"
        else:
            raise RuntimeError(
                "Not sure what numpy.dtype: %s translates to in vtk"
                % (str(arrayDataType))
            )

        xcellvar = ET.Element(
            "DataArray",
            {
                "type": vtkDataType,
                "NumberOfComponents": str(ncomponents),
                "format": datatype,
                "Name": var,
            },
        )
        # string = str()
        if datatype == "binary":
            string = b""
        else:
            string = str()

        for ielem in range(nelems):
            val = cellData[var][ielem, :]
            for icomponent in range(ncomponents):
                if datatype == "binary":
                    string = string + struct.pack(binspec, val[icomponent])
                else:
                    string = string + "".join([repr(val[icomponent]), " "])

        # if datatype=="binary":
        #     string=FSUtils.bincalc(string)

        xcellvar.text = string
        xCellData.append(xcellvar)

    # import pdb;pdb.set_trace()
    outFile = open(vtufilename, "wb")
    if datatype == "binary":
        filestr = ET.tostring(xvtufile)
    else:
        filestr = FSUtils.prettyPrint(xvtufile)

    outFile.write(filestr)

    outFile.close()
