"""Functions to build element connectivities for paraview which supports various types of element"""

import numpy


def normalise_elements(elems, elemName):
    """Give elements a common structure
    "Normal" elements
     -- nodes stored in elem.nodes
     "Spider" elements
     -- dependent node stored in elem.refnode
     -- independent nodes stored in elem.indepnodes"""

    if elemName in ["CBAR", "CBEAM", "RBAR", "CBUSH", "RBAR1"]:
        for elem in elems:
            elem.nodes = [elem.Ga, elem.Gb]
    elif elemName in ["PLOTEL"]:
        for elem in elems:
            elem.nodes = [elem.g1, elem.g2]
    elif elemName in ["CONM2"]:
        for elem in elems:
            elem.nodes = [elem.nid]
    elif elemName in ["RBE3"]:
        for elem in elems:
            elem.refnode = elem.refgrid
            elem.indepnodes = []
            # pdb.set_trace()
            # for WtCG_group in elem.WtCG_groups:
            for WtCG_group in elem.wt_cg_groups:
                elem.indepnodes += WtCG_group[2]
    elif elemName in ["RBE1"]:
        # This is like a spider except that there are possibly multiple reference nodes. Needs some work
        for elem in elems:
            elem.depnodes = elem.Gmi
            elem.indepnodes = elem.Gni
    elif elemName in ["RBE2"]:
        for elem in elems:
            elem.refnode = elem.gn
            elem.indepnodes = elem.Gmi


def sort_normal_element(fem, elemName, elems, nodes, nodeIDs):
    """Sorter for "normal" elements -- i.e. has connectivity that is nice and easily defined."""

    node_in_group = numpy.ones((len(nodes),), dtype=numpy.int_) * -1
    nNodesInElem = len(elems[0].nodes)
    elemtable = numpy.zeros((len(elems), nNodesInElem), dtype=numpy.int_)
    elemIDs = numpy.zeros((len(elems),), dtype=numpy.int_)
    # print "len(nodes)",len(nodes)
    # print NodeIDs.index(4340492), NodeIDs.index(4340502), NodeIDs.index(4340500), NodeIDs.index(4340490)

    print("Building connectivity for %s" % (elemName))
    for i, elem in enumerate(elems):
        # elem.nodes # Nodes in this element
        # print(elem)
        # elemNodeIDs = [ n.Nid() for n in elem.nodes]
        try:
            elemNodeIDs = elem.node_ids
        except AttributeError:
            elemNodeIDs = [n.nid for n in elem.nodes]
        elemIDs[i] = int(elem.eid)
        # print "1",elemNodeIDs
        try:
            nodeidxs = numpy.searchsorted(nodeIDs, elemNodeIDs)
        except TypeError:
            import pdb

            pdb.set_trace()
        # print "2",nodeidxs
        elemtable[i, :] = nodeidxs
        for idx in nodeidxs:
            node_in_group[idx] = 1

    # print elemtable
    # print node_in_group
    # renumber elemtable for nodes not connected to these elements.
    # reuse node_in_group
    count = 0
    for i in range(len(node_in_group)):
        if node_in_group[i] == 1:
            node_in_group[i] = count
            count += 1
        else:
            node_in_group[i] = -1

    print("  connectivity done...")

    return len(elems), elemtable, node_in_group, count, elemIDs


def sort_spider_element(fem, elemName, elems, nodes, nodeIDs):
    """Spider elements are like RBE3s have lots of bar elements (legs) radiating from a central node"""

    # Count the number of bar elements needed
    # There is one for each non-central node
    node_in_group = numpy.zeros((len(nodes),), dtype=numpy.int_)

    nelems = 0
    for elem in elems:
        nelems += len(elem.indepnodes)

    nNodesInElem = 2  # Bar elements
    elemtable = numpy.zeros((nelems, nNodesInElem), dtype=numpy.int_)
    elemIDs = numpy.zeros(
        (nelems,), dtype=numpy.int_
    )  # This references the actual elements not the extras created here.
    # Populate the element table
    ecount = 0
    print("Building connectivity for %s" % (elemName))
    for i, elem in enumerate(elems):
        # pdb.set_trace()
        # elemNodeIDs = [elem.refnode.Nid()]
        elemNodeIDs = [elem.refnode]
        # elemNodeIDs += [n.Nid() for n in elem.indepnodes]
        elemNodeIDs += [n for n in elem.indepnodes]

        # print "nodeids :",elemNodeIDs
        nodeidxs = numpy.searchsorted(nodeIDs, elemNodeIDs)

        # print "nodeidxs:",nodeidxs
        node_in_group[nodeidxs[0]] = 1
        for idx in nodeidxs[1:]:
            elemIDs[ecount] = int(elem.eid)
            elemtable[ecount, :] = [nodeidxs[0], idx]
            ecount += 1
            node_in_group[idx] = 1

    # print elemtable

    # renumber elemtable for nodes not connected to these elements.
    # reuse node_in_group
    count = 0
    for i in range(len(node_in_group)):
        if node_in_group[i] == 1:
            node_in_group[i] = count
            count += 1
        else:
            node_in_group[i] = -1

    print("  connectivity done...")

    return nelems, elemtable, node_in_group, count, elemIDs


def sort_super_spider_element(fem, elemName, elems, nodes, nodeIDs):
    """Super Spider elements like RBE1s have lots of bar elements (legs) radiating from one or more central nodes"""
    # Count the number of bar elements needed
    # There is one for each non-central node
    node_in_group = numpy.zeros((len(nodes),), dtype=numpy.int_)

    nelems = 0
    for elem in elems:
        nelems += len(elem.depnodes) * len(elem.indepnodes)

    nNodesInElem = 2  # Bar elements
    elemtable = numpy.zeros((nelems, nNodesInElem), dtype=numpy.int_)
    elemIDs = numpy.zeros(
        (nelems,), dtype=numpy.int_
    )  # This references the actual elements not the extras created here.
    # Populate the element table
    ecount = 0
    print("Building connectivity for %s" % (elemName))
    for i, elem in enumerate(elems):
        elemNodeIDs = [n.Nid() for n in elem.depnodes]
        elemNodeIDs += [n.Nid() for n in elem.indepnodes]

        # print "nodeids :",elemNodeIDs
        nodeidxs = numpy.searchsorted(nodeIDs, elemNodeIDs)

        ndeps = len(elem.depnodes)
        nindeps = len(elem.indepnodes)

        # print "nodeidxs:",nodeidxs
        node_in_group[nodeidxs[0]] = 1
        for jdx in nodeidxs[0:ndeps]:
            node_in_group[jdx] = 1
            for idx in nodeidxs[ndeps:]:
                elemIDs[ecount] = int(elem.eid)
                elemtable[ecount, :] = [jdx, idx]
                ecount += 1
                node_in_group[idx] = 1

    # print elemtable

    # renumber elemtable for nodes not connected to these elements.
    # reuse node_in_group
    count = 0
    for i in range(len(node_in_group)):
        if node_in_group[i] == 1:
            node_in_group[i] = count
            count += 1
        else:
            node_in_group[i] = -1

    print("  connectivity done...")

    return nelems, elemtable, node_in_group, count, elemIDs


def sort_elems(fem, elemName, elems, nodes, nodeIDs):
    """Switching function -- chose which type of element goes where here"""

    # Normalise elements so they have a consistent structure.
    normalise_elements(elems, elemName)

    if elemName in ["RBE3", "RBE2"]:
        nelems, elemtable, node_in_group, nnodes_in_group, elemIDs = (
            sort_spider_element(fem, elemName, elems, nodes, nodeIDs)
        )
    elif elemName in ["RBE1"]:
        nelems, elemtable, node_in_group, nnodes_in_group, elemIDs = (
            sort_super_spider_element(fem, elemName, elems, nodes, nodeIDs)
        )
    else:
        # Default -- assume anything here is a normal element -- worth at try at least.
        nelems, elemtable, node_in_group, nnodes_in_group, elemIDs = (
            sort_normal_element(fem, elemName, elems, nodes, nodeIDs)
        )

    return nelems, elemtable, node_in_group, nnodes_in_group, elemIDs
