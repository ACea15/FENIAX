import numpy as np


def node_id_dic(lines):
    dmig = 0
    j = 0
    id_list = []
    nodes_list = []
    indices = []
    id_dic = {}
    for i in range(len(lines)):
        if lines[i][0] == "DMIG":
            dmig += 1
            if dmig == 2:
                break

        if lines[i][0] == "DMIG*":
            if lines[i][4] not in nodes_list:
                id_list.append([j, int(lines[i][4])])
                nodes_list.append(lines[i][4])
                j = j + 1
            indices.append([])
            k = 0
        elif lines[i][0] == "*":
            indices[-1].append(k)
            k = k + 1

    for i in range(len(id_list)):
        id_dic[id_list[i][1]] = i
    nDoF = len(id_list) * 6
    return id_dic, id_list, nDoF, indices


def populate_matrices(lines, nDoF, id_dic):
    DMIG = 0
    matrices = []
    dmig = 0
    A = np.zeros((nDoF, nDoF))
    for i in range(len(lines)):
        # pdb.set_trace()
        # for i in range(43660,len(lines)):

        if lines[i][0] == "DMIG*":
            k_node = lines[i][4]
            k_dof = lines[i][5]
        if lines[i][0] == "*":
            col = 6 * (id_dic[int(lines[i][2])]) + int(lines[i][4]) - 1
            row = 6 * (id_dic[int(k_node)]) + int(k_dof) - 1
            globals()[lines[dmig][1]][row, col] = globals()[lines[dmig][1]][
                col, row
            ] = float(lines[i][5])
        if lines[i][0] == "DMIG" or lines[i][0] == "DTI":
            dmig = i
            globals()[lines[dmig][1]] = np.zeros((nDoF, nDoF))
            DMIG += 1
            if DMIG == 3:
                break
            matrices.append(lines[dmig][1])

    return [eval(matrices[i]) for i in range(len(matrices))]


def readfile_split_pch(filename):
    Nastran2011 = 1
    with open(filename, "r") as file1:
        lines = file1.readlines()

    lines_split = []
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        word = []
        for j in range(0, 33, 8):
            word.append(lines[i][j : j + 8].strip())
        if Nastran2011:
            word.append(lines[i][40:57].strip().replace("D", "E"))
        else:
            word.append(lines[i][40:57].strip())
        lines_split.append(word)

    return lines_split


def read_pch(pchfname):
    # Split the pch file into words.
    lines = readfile_split_pch(pchfname)

    # Read ID dictionary
    id_dic, id_list, nDoF, indices = node_id_dic(lines)

    # Populate matrices M and E
    Kaa, Maa = populate_matrices(lines, nDoF, id_dic)

    # print printmat(A)

    return id_list, Kaa, Maa
