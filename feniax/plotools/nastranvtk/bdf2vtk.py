import sys, argparse, os, numpy, pdb
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2

import feniax.plotools.nastranvtk.vtu_writer as vtu_writer
import feniax.plotools.nastranvtk.vtm_writer as vtm_writer
import feniax.plotools.nastranvtk.common as common
import feniax.plotools.nastranvtk.result_prep as result_prep


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


def pointInBox(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    if xmin < x < xmax:
        if ymin < y < ymax:
            if zmin < z < zmax:
                return True
    return False


def write_vtk_vis(filename, fem, onlypts):
    # generate vtk file for plotting over paraview docs
    fout2 = open(filename + ".vtk", "w")
    nodes = []
    for nid, node in sorted(fem.nodes.iteritems()):
        nodes.append(node)

    # Remove any nodes that are miles away from the centre
    # bounding box: xmin,xmax,ymin,ymax,zmin,zmax
    bbox = [0.0, 80.0, -30.0, 30.0, -20.0, 20.0]

    print(len(nodes))

    farawayNodes = numpy.zeros((len(nodes),), dtype=numpy.int_)

    print("***********************")
    print("    FAR AWAY NODES")
    for i, node in enumerate(nodes):
        nodepos = node.get_position()
        if numpy.any(numpy.isnan(nodepos)) or numpy.any(numpy.isinf(nodepos)):
            print(
                "Inf or nan: %10.3f %10.3f %10.3f"
                % (nodepos[0], nodepos[1], nodepos[2])
            )
            farawayNodes[i] = 1

        if not pointInBox(
            nodepos[0],
            nodepos[1],
            nodepos[2],
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            bbox[4],
            bbox[5],
        ):
            farawayNodes[i] = 1
            print(
                "Node Coords: %10.3f %10.3f %10.3f"
                % (nodepos[0], nodepos[1], nodepos[2])
            )

    print("***********************")

    reducednodes = [node for i, node in enumerate(nodes) if farawayNodes[i] == 0]
    nodes = reducednodes

    numNodes = len(nodes)
    print(len(nodes))

    fout2.write("# vtk DataFile Version 2.0\n")
    fout2.write("Deformation lines\n")
    fout2.write("ASCII\n")
    fout2.write("DATASET UNSTRUCTURED_GRID\n")

    # write Points into the file
    fout2.write("POINTS %i float\n" % (numNodes))
    for i in range(numNodes):
        nodepos = nodes[i].get_position()
        fout2.write("%9.7f %9.7f %9.7f\n" % (nodepos[0], nodepos[1], nodepos[2]))

    # creates the cells which contain the Shell information
    # if Quad!=[]:
    #     fout2.write('CELLS %i %i\n' % ((len(Quad)+len(Tria)+len(CBar)),((len(Tria)+len(Quad)+len(CBar))*5)))
    # for i in range(len(Quad)):
    #     fout2.write('4 %4i %4i %4i %4i\n' % (Quad[i].grids[0],Quad[i].grids[1],Quad[i].grids[2],Quad[i].grids[3]))
    # for i in range(len(Tria)):
    #     fout2.write('3 %4i %4i %4i\n' % (Tria[i].grids[0],Tria[i].grids[1],Tria[i].grids[2]))
    # for i in range(len(CBar)):
    #     fout2.write('2 %4i %4i\n' % (CBar[i].grids[0],CBar[i].grids[1]))

    # if Quad==[]:
    fout2.write("CELLS %i %i\n" % (numNodes, numNodes * 2))
    for i in range(numNodes):
        fout2.write("1 %4i\n" % (i))
    fout2.write("\n")

    # creates the cell types for Quad and Tria elements
    # if Quad!=[]:
    #     fout2.write('CELL_TYPES %i\n' % (len(Quad)+len(Tria)+len(CBar)))
    # for j in range(len(Quad)):
    #     fout2.write('%i\n' % (9))
    # for j in range(len(Tria)):
    #     fout2.write('%i\n' % (5))
    # for j in range(len(CBar)):
    #     fout2.write('%i\n' % (3))

    # if Quad==[]:
    fout2.write("CELL_TYPES %i\n" % (numNodes))
    for j in range(numNodes):
        fout2.write("%i\n" % (1))

    fout2.write("\n")

    fout2.write("POINT_DATA %i\n" % (numNodes))
    fout2.write("SCALARS Grid_ID int \n")
    fout2.write("LOOKUP_TABLE default\n")
    for i in range(numNodes):
        fout2.write("%8i \n" % nodes[i].Nid())

    fout2.write("\n")

    # if len(pt[0].disp)>0:
    # # Write displacement Data
    #     for c in range(len(subcasenames)):
    #         subcasenames[c]=subcasenames[c].replace(' ','')
    #         fout2.write('VECTORS '+subcasenames[c]+' float\n')
    #         for i in range(numNodes):
    #             fout2.write('%9.7f %9.7f %9.7f\n' % (pt[i].disp[3*c],pt[i].disp[1+3*c],pt[i].disp[2+3*c]))

    # # Write conm Data
    # for i in range(numNodes):
    #     if pt[i].mass!=0:
    #         fout2.write('VECTORS CONM float\n')
    #         for i in range(numNodes):
    #             fout2.write('%9.7f %9.7f %9.7f  \n' % (0.0,0.0,-pt[i].mass))
    #         break
    # fout2.write('\n')

    fout2.close()


def run(
    bdfname,
    op2name,
    vtkname,
    onlypts,
    punch=False,
    modes2keep=None,
    fileformat="binary",
    xref=True,
):
    fem = BDF(debug=True, log=None)

    # Read bdf
    fem.read_bdf(bdfname, xref=xref, punch=punch)

    print(fem.get_bdf_stats())

    op2 = None
    if op2name is not None:
        op2 = OP2(debug=True, log=None)
        op2.read_op2(op2name)

        elems = result_prep.build_eigvec_elems(fem, op2)
        # print type(elems)
        # print type(fem)
        fem.pseudoElements = elems
        # print type(fem.pseudoElements)

        if modes2keep is not None:
            modes2keep = buildNumRangeFromString(modes2keep)

        # print op2.eigenvectors[i]

    # Write vtk visualisation
    # write_vtk_vis(vtkname,fem,onlypts)

    # Write vtm multiblock file
    vtudir = os.path.join("./", vtkname)
    common.MakeDir(vtudir)

    elementsToWrite = [
        "CBEAM",
        "RBE1",
        "RBE3",
        "RBE2",
        "RBAR",
        "RBAR1",
        "CONM2",
        "CQUAD4",
        "CTRIA3",
        "CBAR",
        "CBUSH",
        "PLOTEL",
    ]

    filesWritten = []
    elemNames = []
    # Stuff to do only once!
    nodes, nodeIDs = vtu_writer.writeVTUPrep(fem)
    for elemName in elementsToWrite:
        fname = vtu_writer.writeVTU(
            vtudir, fem, op2, elemName, nodes, nodeIDs, fileformat, modes2keep
        )

        # elems = [value  for key,value in fem.elements.items() if value.type == 'PLOTEL']
        # pdb.set_trace()

        if fname is not None:
            filesWritten.append(fname)
            elemNames.append(elemName)

    # Write eigenvectors - point data
    if op2name is not None:
        elemName = "EIGVEC"
        fname = vtu_writer.writeVTU(
            vtudir, fem, op2, elemName, nodes, nodeIDs, fileformat, modes2keep
        )
        if fname is not None:
            filesWritten.append(fname)
            elemNames.append(elemName)

    vtm_writer.writeVTM("./", vtkname, filesWritten, elemNames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a NASTRAN .bdf file into a vtk file for visualisation in Paraview"
    )
    subparsers = parser.add_subparsers()
    parser = subparsers.add_parser("Test", help="test")
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="The first argument is the name of the bdf-file, the second argument is an optional op2-file (put PARAM,POST,-1 in your bdf to get nastran to write this file out)",
    )
    parser.add_argument(
        "--onlypts",
        action="store_true",
        default=False,
        help="Read only points -- time saving relative to reading everything",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="", help="Output vtk filename"
    )
    parser.add_argument(
        "-fmt",
        "--fileformat",
        type=str,
        default="binary",
        help='Change vtk output datatype to ascii setting this to "ascii"',
    )
    parser.add_argument(
        "-p",
        "--punch",
        action="store_true",
        default=False,
        help="Is this a punch card, i.e. a partial FEM without the header section? if so use this. ",
    )
    parser.add_argument(
        "-m",
        "--modes2keep",
        type=str,
        default=None,
        help="Limit the number of modes in the output to this range ",
    )

    (args) = parser.parse_args()

    bdfname = args.filenames[0]
    op2name = None
    if len(args.filenames) == 2:
        op2name = args.filenames[1]
    elif len(args.filenames) > 2:
        raise IOError(
            "Specify only one bdf and (optionally) one op2. You specified: %s"
            % str(args.filenames)
        )

    if args.output == "":
        jobname = os.path.splitext(bdfname)[0]
        vtkname = jobname.replace(".bdf", "")
    else:
        jobname = os.path.splitext(args.output)[0]
        vtkname = args.output.replace(".vtk", "")

    run(
        bdfname,
        op2name,
        vtkname,
        args.onlypts,
        args.punch,
        args.modes2keep,
        args.fileformat,
    )
