import numpy


# Functions to prepare results from an f06 for plotting
class PseudoElement:
    """A fake element! To build point only elements for plotting only"""

    def __init__(self, elemtype, ID, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        self.nodes = nodes
        self.eid = ID
        self.type = elemtype


def build_eigvec_elems(fem, f06):
    # Check if there are any eigenvectors in the solution
    elems = {}
    if len(f06.eigenvectors.items()):
        # Need to think about subcase support -- not considered yet.

        eigvals = f06.eigenvectors[1].eigns
        print(eigvals)
        print(type(f06.eigenvectors[1]))
        print(f06.eigenvectors[1])
        print("mode")
        print(f06.eigenvectors[1].mode_cycle)
        # print type(elems)
        for i, eigval in enumerate(eigvals):
            if i == 0:
                # build elements
                nodeIDs = [x[0] for x in f06.eigenvectors[1].node_gridtype]
                # print nodeIDs
                nodeIDs.sort()
                nodes = [fem.Node(nid) for nid in nodeIDs]

                # Calculate a characteristic length for the FEM (the average absolute coordinate in all directions)
                length = 0
                length2 = 0
                for node in nodes:
                    length += numpy.sum(numpy.abs(node.get_position()))
                length = length / (3.0 * len(nodes))

                rejectNodes = []

                # A point is rejected if any of its coordinates are greater than 10 times this length
                for inode, node in enumerate(nodes):
                    # If the node is faraway ignore it
                    if any([abs(v) > 10 * length for v in node.get_position()]):
                        rejectNodes.append(node)
                        pass  # Dont create a node
                    else:
                        elems[inode] = PseudoElement("EIGVEC", nodeIDs[inode], node)

                break

        if len(rejectNodes) > 0:
            print()
            print("INFO: The following nodes were not included in the EIGVEC output")
            print("      because they were too far away")
            print("NOTE: Nodes are rejected if they have any coordinate that is more")
            print("10 times the FEM average coordinate")
            print()
            print("  FEM average coordinate: %f" % length)
            print("  %12s | Coordinates (global system)" % ("ID"))
            for node in rejectNodes:
                print(
                    "  %12i | %15.10E %15.10E %15.10E"
                    % (
                        node.nid,
                        node.Position()[0],
                        node.Position()[1],
                        node.Position()[2],
                    )
                )

    return elems
