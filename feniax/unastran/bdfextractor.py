from pyNastran.bdf.bdf import BDF
from dataclasses import dataclass
import numpy as np


@dataclass
class RBE3:
    elements: list
    independent_nodes: list[int]
    dependent_nodes: list[int]
    dinpendent_link: list[list[int]]
    inodes_coord: np.array
    dnodes_coord: np.array
    inodes_map: dict[int:int]
    # dnodes_map: dict[int:int]
    arm_coord: np.array = None

    def __post_init__(self):
        arm = []
        for i, niref in enumerate(self.dependent_nodes):
            dnode_coord = self.dnodes_coord[i]
            for j, nj in enumerate(self.dinpendent_link[i]):
                inode_index = self.inodes_map[nj]
                inode_coord = self.inodes_coord[inode_index]
                arm.append(inode_coord - dnode_coord)
        self.arm_coord = np.array(arm)


def extract_RBE3(bdf, element):
    inodes = element.independent_nodes
    dnode = element.refgrid  # element.dependent_nodes
    inodes_coord = [bdf.nodes[i].get_position() for i in inodes]
    dnodes_coord = [bdf.nodes[dnode].get_position()]
    return inodes, dnode, inodes_coord, dnodes_coord


def build_RBE3(bdf: BDF) -> RBE3:
    independent_nodes = []
    independent_nodes_map = dict()
    dependent_nodes = []
    dinpendent_link = []
    independent_nodes_coord = []
    dependent_nodes_coord = []
    rbe3_elems = []
    for k, v in bdf.rigid_elements.items():
        if v.type == "RBE3":
            inodes, dnode, inodes_coord, dnodes_coord = extract_RBE3(bdf, element=v)
            rbe3_elems.append(k)
            dependent_nodes.append(dnode)
            dependent_nodes_coord += dnodes_coord
            link = []
            for ini, inicoord in zip(inodes, inodes_coord):
                if ini not in independent_nodes:
                    independent_nodes_map[ini] = len(independent_nodes)
                    independent_nodes.append(ini)
                    link.append(ini)
                    independent_nodes_coord.append(inicoord)
            dinpendent_link.append(link)

    return RBE3(
        rbe3_elems,
        independent_nodes,
        dependent_nodes,
        dinpendent_link,
        np.array(independent_nodes_coord),
        np.array(dependent_nodes_coord),
        independent_nodes_map,
    )


def iterate_rigidelements(bdf: BDF, ktype):
    container = []
    builder = globals()[f"build_{ktype.upper()}"]
    for k, v in bdf.rigid_elements.items():
        if v.type == ktype:
            container.append(builder(bdf, element=v, ID=k))
    return container


if __name__ == "__main__":
    bdf = BDF()
    bdf.read_bdf(
        "/media/acea/work/projects/FEM4INAS/examples/wingSP/NASTRAN/wing400d.bdf"
    )
    # rbe3 = iterate_rigidelements(bdf, "RBE3")
    rbe3 = build_RBE3(bdf)
