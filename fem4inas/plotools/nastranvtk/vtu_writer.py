# File to write a binary vtu file for paraview. -- these are the parts of the multiblock data set in a vtm file. 
import xml.etree.ElementTree as ET
import sys
import fem4inas.plotools.nastranvtk.GenericVTKWriter.VTUWriter as VTUWriter


import numpy, os

import fem4inas.plotools.nastranvtk.elem_sorter as elem_sorter
import fem4inas.plotools.nastranvtk.elem_data as elem_data
import fem4inas.plotools.nastranvtk.common as common
import fem4inas.plotools.nastranvtk.point_data as point_data

def writeVTUPrep(fem):
    
    nodes = fem.nodes.values()
    nodes = sorted(nodes,key=lambda x:x.Nid())
    NodeIDs = [node.Nid() for node in nodes] # Still sorted 
    
    return nodes, NodeIDs

def writeVTU(filepath,fem,op2,elemName,nodes,nodeIDs,datatype="binary",modes2keep=None):
    '''Code to write an vtu file containing only elemName elements'''
    
    # Grab the elements

    if elemName in ["RBE1","RBE2","RBE3","RBAR","RBAR1"]:
        elems = [value  for key,value in fem.rigid_elements.items() if value.type == elemName]
        #elemIDsI = [key  for key,value in fem.rigidElements.items() if value.type == elemName]
    elif elemName in ['CONM1','CONM2','CMASS1','CMASS2','CMASS3','CMASS4']:
        elems = [value  for key,value in fem.masses.items() if value.type == elemName]
    elif elemName in ["EIGVEC"]:
        elems = [value  for key,value in fem.pseudoElements.items() if value.type == elemName]
        #elemIDsI = [key  for key,value in fem.pseudoElements.items() if value.type == elemName]
    else:
        elems = [value  for key,value in fem.elements.items() if value.type == elemName]
        #elemIDsI = [key  for key,value in fem.elements.items() if value.type == elemName]
        
    print("Found %i %s elements" % (len(elems),elemName))
    vtufilename = None
    if len(elems) > 0:

        nelems,elemtable,node_in_group,nnodes_in_group,elemIDs = elem_sorter.sort_elems(fem,elemName,elems,nodes,nodeIDs)
        point_data = {}
        cell_data = {}
        
        # build points table
        points = numpy.zeros((nnodes_in_group,3))

        point_data["GRID_ID"] = numpy.zeros((nnodes_in_group,),dtype=int)

        cell_data["ELEM_ID"] = elemIDs

        inode=0
        for i,node in enumerate(nodes):
            if node_in_group[i] != -1:
                nodepos = node.get_position()
                points[inode,:] = nodepos
                point_data["GRID_ID"][inode] = node.Nid()
                inode+=1
        
        # connectivity - in elemtable
        # Renumber elemtable
        for i in range(len(elemtable)):
            for j in range((len(elemtable[i]))):
                if node_in_group[elemtable[i][j]] != -1:
                    elemtable[i][j] = node_in_group[elemtable[i][j]]
                else:
                    raise RuntimeError("Something wrong in connectivity calculation. This node is not in the group but is in the connectivity table")
                

        # point_data

        # Mode shapes (eigenvectors)
        if op2 is not None:
            if len(op2.eigenvectors.keys()) > 0:
                eigvals = op2.eigenvectors[1].eigns
                # Use the cycles if they are there otherwise calculate them
                if hasattr(op2.eigenvectors[1],"mode_cycle") and not(isinstance(op2.eigenvectors[1].mode_cycle,float)):
                    cycles = op2.eigenvectors[1].mode_cycle
                else:
                    cycles = numpy.sqrt(numpy.abs(numpy.array(eigvals)))/(2.0*numpy.pi)
                print(f"cycles: {cycles}")
                for i,eigval in enumerate(eigvals):
                    
                    if i == 0:
                        # Index all the EigvecNodeIDs in nodeIDs
                        EigvecNodeIDs =  [ x[0] for x in op2.eigenvectors[1].node_gridtype]
                        print(len(EigvecNodeIDs),len(nodeIDs))
                        if len(EigvecNodeIDs) == len(nodeIDs):
                            nodeindexes = range(len(nodeIDs))
                        else:
                            print ("Indexing eigenvector nodes...")
                            nodeindexes =  numpy.searchsorted(nodeIDs,EigvecNodeIDs) # Index in global nodeID list
                            

                    if modes2keep is None or i+1 in modes2keep:
                        print ("Building mode %i: %.4f Hz" % (i+1,cycles[i]))
                        name = "mode %03i: %.4f Hz" % (i+1,cycles[i]) 
                        point_data[name] = numpy.zeros((nnodes_in_group,3))

                      
                        string = str()

                        print("Copying eigenvector data...")
                        for inodeID in range(len(EigvecNodeIDs)): # Index in eigenvector table
                            if node_in_group[nodeindexes[inodeID]] != -1:
                                ielemnode = node_in_group[nodeindexes[inodeID]]
                                outcoord = fem.Node(EigvecNodeIDs[inodeID]).Cd()
                                if isinstance(outcoord,int) and outcoord == 0:
                                    point_data[name][ielemnode,:] = op2.eigenvectors[1].data[i,inodeID,0:3]
                                elif isinstance(outcoord,int):
                                    #print "This coordinate system with ID %i is not cross-referenced in node %i" % (outcoord,EigvecNodeIDs[inodeID])
                                    outcoord = fem.Coord(outcoord)
                                    point_data[name][ielemnode,:] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,inodeID,0:3])
                                    #raise RuntimeError("This coordinate system with ID %i is not cross-referenced in node %i" % (outcoord,EigvecNodeIDs[inodeID]))
                                else:
                                    point_data[name][ielemnode,:] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,inodeID,0:3])
                                    
                                #print EigvecNodeIDs[inodeID], op2.eigenvectors[1].data[i,inodeID,0:3],point_data[name][ielemnode,:]

        if elemName in ["CONM2"]:
            cell_data["Mass"] = numpy.zeros((nelems,),dtype=float)
            for ielemID,elemID in enumerate(elemIDs):
                #print elemID
                elem = fem.masses[elemID]
                cell_data["Mass"][ielemID]= elem.mass
                
                                

        print("Writing",elemName + ".vtu")
        vtufilename=os.path.join(filepath,elemName + ".vtu")
        VTUWriter.WriteVTU(vtufilename,points,elemtable,point_data,cell_data,datatype=datatype)

   
    return vtufilename
