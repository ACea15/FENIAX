'''Functions to extract point data for plotting in Paraview '''
import xml.etree.ElementTree as ET
import numpy, struct
from fem4inas.plotools.nastranvtk.GenericVTKWriter.Common import bincalc

def extract_point_data(xPointData,nodes,node_in_group,op2,elemName,datatype):
    '''Extract point specific data - switcher function'''
    
    if elemName in ["EIGVEC", "PLOTEL"]:
        # pdb.set_trace()
        if op2 is not None:
            if len(op2.eigenvectors.keys()) > 0:
                eigvals = op2.eigenvectors[1].eigrs
                # Use the cycles if they are there otherwise calculate them
                if hasattr(op2.eigenvectors[1],"mode_cycle") and not(isinstance(op2.eigenvectors[1].mode_cycle,int)):
                    cycles = op2.eigenvectors[1].mode_cycle
                else:
                    cycles = numpy.sqrt(numpy.real(numpy.array(eigvals)))/(2.0*numpy.pi)

                print("cycles",cycles)

                for i,eigval in enumerate(eigvals):

                    nodeIDs =  [ x[0] for x in op2.eigenvectors[1].node_gridtype]
                    #nodeIDs.sort() # not really sure you want to do this. 

                    string = str()
                    for inodeID in range(len(nodeIDs)):
                        translations = op2.eigenvectors[1].data[i,inodeID,0:3]
                        if datatype=="binary":
                            string+=struct.pack('<fff',translations[0],translations[1],translations[2])
                        else:
                            string = string + ''.join([repr(translations[0]),' ',repr(translations[1]),' ',
                                                       repr(translations[2]),' '])
                    if datatype=="binary":
                        string=bincalc(string)


                    name = "mode %03i: %.4f Hz" % (i+1,cycles[i]) 

                    xeigvec = ET.Element("DataArray",{"type":"Float32",
                                                      "NumberOfComponents":"3",
                                                      "format":datatype,
                                                      "Name":name})

                    xeigvec.text = string
                    xPointData.append(xeigvec)



    pass 
