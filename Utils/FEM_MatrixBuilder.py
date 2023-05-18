import sys, argparse, os, numpy, pdb
import scipy.linalg, scipy.sparse.linalg,scipy.spatial
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
from pyNastran.op4.op4 import OP4
sys.path.append(os.getcwd())
#import utils
pathtothismodule =os.getcwd()
sys.path.insert(0,os.path.abspath(os.path.join(pathtothismodule,"../")))
#sys.path.append("/shared/programs/utils/FEM_tools/pyNastran/pyNastran-0.7.2")
#sys.path.append("/pyNastran")
#from pyNastran.f06.f06 import F06 # Not supported by pyNastran-0.7.2
#sys.path.append("/home/ac5015/Downloads/pyNastran-master")

#pathtothismodule2=os.path.dirname(__file__)

#from Nastran2FlowSim_modified import read_pch
#from CalculateModalInertia import GetEigenvectors
#sys.path.insert(0,os.path.abspath(os.path.join(pathtothismodule,"../FlexMB")))
#from Common import printmat, mdot


from collections import OrderedDict






#=======================================================================================================================================================

def mdot(*args):
    '''Multiple dot function '''
    if len(args) == 2:
        return numpy.dot(args[0],args[1])
    elif len(args) < 2:
        raise IOError("mdot needs at least two arguments")
    else:
        return numpy.dot(args[0],mdot(*args[1:]))

def printmat(mat,name=None,fmt="%22.15g",tol=None):

    s = mat.shape
    if len(s) == 1:
        mat = mat.reshape((s[0],1)).copy()

    ni,nj = mat.shape

    matstr = ""
    if name is not None:
        matstr = "%s\n" % name

    for i in range(ni):
        for j in range(nj):
            num = mat[i,j]
            if tol is not None:
                if abs(mat[i,j]) < tol:
                    num = 0
            matstr += fmt % num
        matstr += "\n"
    return matstr



def skew(u):
    '''Puts a vector into skew symmetric form '''
    return numpy.array([[0.0,-u[2],u[1]],[u[2],0.0,-u[0]],[-u[1],u[0],0.0]])
##=====================================================================================================================================================
def write_structuralGrid_file(fname,structgrid):

    nnodes = len(structgrid[:,0])

    fd2 = open(fname,'w')
    fd2.write("""TITLE = \"Reduced Structural Grid\"
VARIABLES = \"x\" \"y\" \"z\" \"id\" \"familynum\" \"forcenode\" \"displnode\" \"body\"
ZONE T=\"Structural nodes\" I=%i , J=1, K=1, F=POINT\n""" % (nnodes))
    for i in range(nnodes):
        fd2.write("%21.15e %21.15e %21.15e %i %i %i %i %i\n" % (structgrid[i,0],structgrid[i,1],structgrid[i,2],structgrid[i,3],1,1,1,1))

    fd2.close()


def write_structuralModes_file(fname,Matrix,nnodes,nmodes,titlestr,matname):

    # Matrix shape: [nmodes x nnodes*6]

    Matrix1 = numpy.reshape(Matrix,(nmodes*nnodes,6))

    fd2 = open(fname,'w')

    fd2.write("""TITLE = \"%s\"
VARIABLES = \"%s\"
ZONE T=\"%s\" I=%i , J=1, K=1, F=BLOCK\n""" % (titlestr,matname,matname,nnodes*nmodes*6))
    for i in range(nnodes*nmodes):
        for j in range(6):
            fd2.write("%21.15e " % (Matrix1[i,j]))
        fd2.write("\n")
    fd2.write("""ZONE T=\"dofs\" I= 6 , J=1, K=1, F=BLOCK
1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00""")

    fd2.close()


def write_structuralMatrix_file(fname,Matrix,titlestr,matname):

    dof = 6
    nnodes = Matrix.shape[0]/dof

    Matrix1 = numpy.reshape(Matrix,(nnodes*nnodes*dof,dof))

    fd2 = open(fname,'w')

    fd2.write("""TITLE = \"%s\"
VARIABLES = \"%s\"
ZONE T=\"%s\" I=%i , J=1, K=1, F=BLOCK\n""" % (titlestr,matname,matname,nnodes*nnodes*6*6))
    for i in range(nnodes*nnodes*6):
        for j in range(6):
            fd2.write("%21.15e " % (Matrix1[i,j]))
        fd2.write("\n")
    fd2.write("""ZONE T=\"dofs\" I= 6 , J=1, K=1, F=BLOCK
1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00  1.000000000000E+00""")

    fd2.close()


def write_structuralVars_file(fname,nnodes):

    fd2 = open(fname,'w')

    fd2.write("""TITLE = \"Structural displacements and forces -- put jig2flight deflections in here.\"
VARIABLES = \"v1\" \"v2\" \"v3\" \"v4\" \"v5\" \"v6\" \"id\" \"familynum\"
ZONE T=\"Structural displacements\" I=%i , J=1, K=1, F=POINT\n""" % (nnodes))
    for i in range(nnodes):
        for j in range(8):
            fd2.write("%21.15e " % (0.0))
        fd2.write("\n")
    fd2.write("""ZONE T=\"Forces\" I=%i , J=1, K=1, F=POINT\n""" % (nnodes))

    for i in range(nnodes):
        for j in range(8):
            fd2.write("%21.15e " % (0.0))
        fd2.write("\n")


    fd2.close()



#=======================================================================================================================================================

def GetEigenvectors(fem,op2,nodeIDs,modes2keep):
    if op2 is not None:
        if len(op2.eigenvectors.keys()) > 0:
            eigvals = op2.eigenvectors[1].eigrs
            istore = 0
            for i,eigval in enumerate(eigvals):

                if i == 0:
                    # Index all the EigvecNodeIDs in nodeIDs
                    EigvecNodeIDs =  [ x[0] for x in op2.eigenvectors[1].node_gridtype]
                    nmodes = len(eigvals)
                    if modes2keep is not None:
                        mask = numpy.ones(len(eigvals),dtype=int)
                        for ii in range(len(mask)):
                            if ii+1 not in modes2keep:
                                mask[ii] = 0
                        nmodes = numpy.sum(mask)

                    Eigvectors = numpy.zeros((nmodes,len(nodeIDs),6))
                    Eigvalues = numpy.zeros((nmodes,1))

                    print len(EigvecNodeIDs),len(nodeIDs)
                    if len(EigvecNodeIDs) == len(nodeIDs):
                        nodeindexes = range(len(nodeIDs))
                    else:
                        print "Indexing eigenvector nodes..."
                        nodeindexes =  numpy.searchsorted(EigvecNodeIDs,nodeIDs) # Index in Eigenvector nodeID list - will break if eigenvectors dont have this node.
                if modes2keep is None or i+1 in modes2keep:
                    print "Building mode %i" % (i+1)
                    Eigvalues[istore] = eigval

                    #print "Copying eigenvector data..."
                    for inodeID in range(len(nodeindexes)):
                        iEigVec = nodeindexes[inodeID] # Index in eigenvector table
                        outcoord = fem.Node(EigvecNodeIDs[iEigVec]).Cd()

                        if isinstance(outcoord,int) and outcoord == 0:
                            Eigvectors[istore,inodeID,0:6] = op2.eigenvectors[1].data[i,iEigVec,0:6]
                        elif isinstance(outcoord,int):
                                  #print "This coordinate system with ID %i is not cross-referenced in node %i" % (outcoord,EigvecNodeIDs[inodeID])
                            outcoord = fem.Coord(outcoord)
                            Eigvectors[istore,inodeID,0:3] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,iEigVec,0:3])
                            Eigvectors[istore,inodeID,3:6] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,iEigVec,3:6])
                        else:
                            Eigvectors[istore,inodeID,0:3] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,iEigVec,0:3])
                            Eigvectors[istore,inodeID,3:6] = outcoord.transform_vector_to_global(op2.eigenvectors[1].data[i,iEigVec,3:6])
                    istore += 1
    #EigvecNodeIDs = numpy.array(EigvecNodeIDs)

    return Eigvectors, Eigvalues


def ExtractEigenvectors(fem,f06):
    '''Get eigenvectors out of the fem and f06 and the points at which the eigenvectors are defined. '''


    eigvals = f06.eigenvectors[1].eigenvalues()
    for i,eigval in enumerate(eigvals):

        if i == 0:

            nodeIDs = f06.eigenvectors[1].translations[i+1].keys()
            nodeIDs.sort()
            nodes = [fem.Node(nid) for nid in nodeIDs]
            nElasticNodes = len(nodes)
            u_bar_0 = numpy.zeros((nElasticNodes,3))
            nModes = len(eigvals)
            phi = numpy.zeros((nModes,6*nElasticNodes))

        for inode,nodeID in enumerate(nodeIDs):
            u_bar_0[inode,:] = nodes[inode].get_position()
            translations = f06.eigenvectors[1].translations[i+1][nodeID]
            rotations = f06.eigenvectors[1].rotations[i+1][nodeID]

            phi[i,inode*6:(inode+1)*6] = numpy.hstack((translations,rotations))


    return phi,u_bar_0,nodeIDs,numpy.array(eigvals)



def EnforceZeroEigvals(nnodes,eigvalsort,eigvecsort):
    '''Use the eigen decomposition theorem to force the rigid body
    modes to have a frequency of zero. Idea nicked from Giovanna
    Ferraro'''
    print "#######################################################"
    print "ENFORCING ZERO EIGENVALUES FOR RIGIDBODY MODES"

    # # Sort eigenvals - these are normalised so Mm = I
    # eigvalsort = numpy.sort(eigval,kind='mergesort')

    # # sort eigenvectors
    # sortinds = numpy.argsort(eigval,kind='mergesort')
    # eigvecsort = numpy.take(eigvec,sortinds,axis=1)

    # # Check I got the take command right
    # for ind in range(10):
    #     print printmat(eigvecsort[:,ind].reshape((nnodes,6)),"Eigenvector No %i - Eigenvalue: %f" % (ind+1,eigval[sortinds[ind]]),"%12.5g")

    # Test eigenvectors are orthogonal - they arent!
    print "Test input eigenvectors are orthogonal"
    test = mdot(eigvecsort,eigvecsort.T)
    print printmat(test,"eigvec orth test (should be identity matrix)","%12.5g",1e-10)


    # Set first 6 eigenvalues to zero
    Km = numpy.diag(eigvalsort)
    for i in range(6):
        Km[i,i] = -1e-9 - i*1e-9 # small number rather than zero?

    KssNew = mdot(eigvecsort,Km,numpy.linalg.inv(eigvecsort))
    MssNew = mdot(eigvecsort,numpy.eye(nnodes*6),numpy.linalg.inv(eigvecsort))
    # get new eigenvalues and eigenvectors (should be very similar except for RB modes)
    eigvalnew,eigvecnew = scipy.linalg.eigh(KssNew,MssNew)

    # Sort eigenvectors again
    # Sort eigenvals - these are normalised so Mm = I
    eigvalnewsort = numpy.sort(eigvalnew,kind='mergesort')
   # sort eigenvectors
    sortinds = numpy.argsort(eigvalnew,kind='mergesort')
    eigvecnewsort = numpy.take(eigvecnew,sortinds,axis=1)

    # Renormalise!
    print "Testing New eigenvectors"
    M = mdot(numpy.linalg.inv(eigvecnewsort),MssNew,eigvecnewsort)
    print printmat(M,"modalMass","%12.5g",1e-6)
    M = mdot(numpy.linalg.inv(eigvecnewsort),KssNew,eigvecnewsort)
    print printmat(M,"modalStiffness","%12.5g",1e-6)

    #eigvecnew = numpy.sqrt(numpy.diag(numpy.linalg.inv(numpy.diag(numpy.diag(M)))))*eigvecnew

    print printmat(numpy.hstack((eigvalsort.reshape(len(eigvalsort),1),eigvalnewsort.reshape(len(eigvalsort),1))),"Old and New Eigenvalues","%12.5g")

    print "NEW EIGENVECTORS"
    for ind in range(10):
        print printmat(eigvecnewsort[:,ind].reshape((nnodes,6)),"Eigenvector No %i - Eigenvalue: %f" % (ind+1,eigvalnew[ind]),"%12.5g")



    print "#######################################################"
    return eigvalnewsort,eigvecnewsort




#=======================================================================================================================================================

def readfile_split_pch(filename):

    Nastran2011=1
    with  open(filename,'r') as file1:
     lines = file1.readlines()

    lines_split=[]
    for i in range(len(lines)):
	lines[i] = lines[i].strip('\n')
	word = []
        for j in range(0,33,8):
           word.append(lines[i][j:j+8].strip())
        if Nastran2011:
	 word.append(lines[i][40:57].strip().replace('D','E'))
        else:
         word.append(lines[i][40:57].strip())
        lines_split.append(word)

    return lines_split


def node_id_dic(lines):
    dmig=0
    j=0
    id_list = []
    nodes_list=[]
    indices=[]
    id_dic={}
    for i in range(len(lines)):

      if lines[i][0]=='DMIG':
       dmig+=1
       if  dmig==2:
        break

      if lines[i][0]=='DMIG*':
       if lines[i][4] not in nodes_list:
        id_list.append([j,int(lines[i][4])])
        nodes_list.append(lines[i][4])
        j=j+1
       indices.append([])
       k=0
      elif lines[i][0]=='*':
       indices[-1].append(k)
       k=k+1

    for i in range(len(id_list)):
	id_dic[id_list[i][1]] = i
    nDoF=len(id_list)*6
    return id_dic,id_list,nDoF,indices

def populate_matrices(lines,nDoF,id_dic):
    DMIG=0
    matrices=[]
    dmig=0
    A=numpy.zeros((nDoF,nDoF))
    for i in range(len(lines)):
    #pdb.set_trace()
    #for i in range(43660,len(lines)):

        if lines[i][0]=='DMIG*':
         k_node=lines[i][4]
         k_dof=lines[i][5]
        if lines[i][0]=='*':
         col=6*(id_dic[int(lines[i][2])])+int(lines[i][4])-1
         row=6*(id_dic[int(k_node)])+int(k_dof)-1
         globals()[lines[dmig][1]][row,col]= globals()[lines[dmig][1]][col,row] =float(lines[i][5])
        if lines[i][0]=='DMIG' or lines[i][0]=='DTI':
          dmig=i
          globals()[lines[dmig][1]]=numpy.zeros((nDoF,nDoF))
          DMIG+=1
          if  DMIG==3:
           break
          matrices.append(lines[dmig][1])

    return [eval(matrices[i]) for i in range(len(matrices))]

def write8(st):
    string = str(st)
    space=8-len(string)
    if space<0:
        raise ValueError('string>8')
    s0=' '
    s=space*s0
    return string+s

def write16(st):
    string=str(st)
    space=16-len(string)
    if space<0:
        raise ValueError('string>16')
    s0=' '
    s=space*s0
    return string+s

def write_num16(st):

    strin = '{:.9e}'.format(st)
    string = list(strin)
    string[-4]='E'
    return ''.join(string)

def write_num8(st):

    strin = '{:.2e}'.format(st)
    string = list(strin)
    #string[-4]='E'
    string[-4]=''
    return ''.join(string)


def write_fem(fname,Mname,M,aset):

    nM=numpy.shape(M)[0]
    assert len(aset)*6==nM
    fw=open(fname,"a")
    fw.write(write8('DMIG'))
    fw.write(write8(Mname))
    fw.write(write8('0'))
    fw.write(write8('6'))
    fw.write(write8('2'))
    fw.write(write8('0'))
    fw.write(write8(''))
    fw.write(write8(''))
    fw.write(write8('%s'%nM))
    fw.write('\n')
    for j in range(nM):
        jn=j/6;jr=j%6
        fw.write(write8('DMIG*'))
        fw.write(write16(Mname))
        fw.write(write16(aset[jn]))
        fw.write(write16(jr+1))
        fw.write('\n')
        for i in range(j+1):
            if M[i,j]!=0:
                inn=i/6;irr=i%6
                fw.write(write8('*'))
                fw.write(write16(aset[inn]))
                fw.write(write16(irr+1))
                fw.write(write_num16(M[i,j]))
                fw.write('\n')

    fw.close()


def write_fem2(fname,Mname,M,aset):

    nM=numpy.shape(M)[0]
    assert len(aset)*6==nM
    fw=open(fname,"a")
    fw.write(write8('DMIG'))
    fw.write(write8(Mname))
    fw.write(write8('0'))
    fw.write(write8('6'))
    fw.write(write8('2'))
    fw.write(write8('0'))
    fw.write(write8(''))
    fw.write(write8(''))
    fw.write(write8('%s'%nM))
    fw.write('\n')
    for j in range(nM):
        jn=j/6;jr=j%6
        fw.write(write8('DMIG'))
        fw.write(write8(Mname))
        fw.write(write8(aset[jn]))
        fw.write(write8(jr+1))
        fw.write('\n')
        for i in range(j+1):
            if M[i,j]!=0:
                inn=i/6;irr=i%6
                fw.write(write8('+'))
                fw.write(write8(aset[inn]))
                fw.write(write8(irr+1))
                fw.write(write_num8(M[i,j]))
                fw.write('\n')

    fw.close()

def write_fem3(Mname,M,aset):

    nM=numpy.shape(M)[0]
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    for j in range(nM):
        jn=j/6;jr=j%6
        for i in range(j+1):
            if M[i,j]!=0:
                inn=i/6;irr=i%6
                GCj.append([aset[jn],jr+1])
                GCi.append([aset[inn],irr+1])
                GM.append(M[i,j])
    #print GM
    numpy.save(Mname+'_GCj.npy',numpy.asarray(GCj))
    numpy.save(Mname+'_GCi.npy',numpy.asarray(GCi))
    numpy.save(Mname+'_GM.npy',numpy.asarray(GM))



# M=numpy.eye(18)
# M[0,3]=5.
# M[2,17]=-1.
# M[9,16]=6

# fname='try1.tex'
# Mname='KAAX'
# aset=[1001,1002,1003]
# write_fem(fname,Mname,M,aset)


# DMIG=0
# matrices=[]
# dmig=0
# A=numpy.zeros((nDoF,nDoF))
# for i in range(len(lines)):
# #pdb.set_trace()
# #for i in range(43660,len(lines)):

#     if lines[i][0]=='DMIG*':
#      k_node=lines[i][4]
#      k_dof=lines[i][5]
#     if lines[i][0]=='*':
#      col=6*(id_dic[int(lines[i][2])])+int(lines[i][4])-1
#      row=6*(id_dic[int(k_node)])+int(k_dof)-1
#      globals()[lines[dmig][1]][row,col]= globals()[lines[dmig][1]][col,row] =float(lines[i][5])
#     if lines[i][0]=='DMIG' or lines[i][0]=='DTI':
#       dmig=i
#       globals()[lines[dmig][1]]=numpy.zeros((nDoF,nDoF))
#       DMIG+=1
#       if  DMIG==3:
#        break
#       matrices.append(lines[dmig][1])




# import pdb

# nDoF=120
# #pdb.set_trace()
# DMIG=0
# matrices=[]
# dmig=0
# A=numpy.zeros((nDoF,nDoF))
# for i in range(len(lines)):

#     if lines[i][0]=='DMIG*':
#      k_node=lines[i][4]
#      k_dof=lines[i][5]
#     if lines[i][0]=='*':
#      col=6*(id_dic[int(lines[i][2])])+int(lines[i][4])-1
#      row=6*(id_dic[int(k_node)])+int(k_dof)-1
#      A[row,col]=A[col,row]=float(lines[i][5])
#     if lines[i][0]=='DMIG': #or lines[i][0]=='DTI':
#       globals()[lines[dmig][1]]=A
#       dmig=i
#       DMIG+=1
#       if  DMIG==5:
#        break
#       matrices.append(lines[dmig][1])
#       A=numpy.zeros((nDoF,nDoF))


def read_pch(pchfname):

    # Split the pch file into words.
    lines = readfile_split_pch(pchfname)

    # Read ID dictionary
    id_dic,id_list,nDoF,indices = node_id_dic(lines)

    # Populate matrices M and E
    Kaa,Maa = populate_matrices(lines,nDoF,id_dic)

    # print printmat(A)

    return id_list,Kaa,Maa

#=======================================================================================================================================================







def LineUpCONM2s(fem,id_list):
    # Map id_list with CONM2 node ids - typically they are different nodes

    # Extract all the conm2s
    conm2s = [value  for key,value in fem.masses.items() if value.type == "CONM2"]
    conm2nodes = [fem.Node(x.Nid()) for x in conm2s]
    conm2onpchnode = numpy.ones((len(conm2s),),dtype=int)*-1
    pchnodeHasConm2 = numpy.zeros((len(id_list),))

    # Use id_list and conm2ids to identify pairs of coincident nodes
    pchnodes = [fem.Node(x[1]) for x in id_list]
    nmatchedconm2s = 0
    for ipchnode,pchnode in enumerate(pchnodes):
        for iconm2,conm2node in enumerate(conm2nodes):
            pnodepos = pchnode.get_position()
            cnodepos = conm2node.get_position()
            if ( pnodepos[0] == cnodepos[0] and
                 pnodepos[1] == cnodepos[1] and
                 pnodepos[2] == cnodepos[2]):
                #matchedconm2s.append(conm2s[iconm2])
                conm2onpchnode[iconm2] = ipchnode
                pchnodeHasConm2[ipchnode] = True
                nmatchedconm2s += 1
                break


    # print "raw   :",[x.Nid() for x in conm2s]
    # print "sorted:",[x.Nid() for x in matchedconm2s]
    # print "len(matchedconm2s) == len(pchnodes):",len(matchedconm2s),len(id_list)

    #  lonelyconm2s = []
    if len(conm2s) > nmatchedconm2s:
        # Lump masses together - move masses not on pchnode to their nearest pchnode
        # Build KDTree
        coords = numpy.zeros((len(pchnodes),3))
        for inode,pchnode in enumerate(pchnodes):
            coords[inode,:] = pchnode.get_position()

        leafsize = 100
        tree = scipy.spatial.cKDTree(coords,leafsize)

        for iconm2,conm2 in enumerate(conm2s):
            if conm2onpchnode[iconm2] == -1: # not on a pchnode
                # Find nearest pch node
                d1,iNearest = tree.query(conm2nodes[iconm2].get_position(),k=1,distance_upper_bound=1000000)
                # Store node
                #lonelyconm2s.append([conm2,iNearest])
                if iNearest < len((pchnodes)):
                    conm2onpchnode[iconm2] = iNearest
                    pchnodeHasConm2[iNearest] = True
                else:
                    raise RuntimeError("Could not find nearest ASET node for this conm2 (%s)" % (conm2nodes[iconm2].get_position()))
    #if len(pchnodes) > len(conm2s):
    #    raise RuntimeError("There are more ASET nodes (%i) than conm2s (%i). This means that not every node will have mass and the mass matrix will be singular." % (len(pchnodes),len(conm2s)))

    #if not numpy.all(pchnodeHasConm2):
    #    print "Nodes with no associated conm2:",[x for i,x in enumerate(id_list) if not pchnodeHasConm2[i]]
    #    raise RuntimeError("Not all pchnodes have an associated conm2 -- mass matrix will be singular - consider removing the nodes listed above from the ASET.")

    #assert(len(matchedconm2s)+len(lonelyconm2s) == len(conm2s))

    return conm2onpchnode,conm2s,pchnodes


def ExtractCONM2Data(fem,conm2s,pchnodes,conm2onpchnode):


    # These conm2s represent all the mass and inertia of the FEM - TODO test for this by checking density == 0.0 in MAT cards.
    #q = [R theta qf]
    totalMass = numpy.zeros((3,3)) # m_RR
    totalInertia = numpy.zeros((3,3))  # m_theta_theta
    crossInertia = numpy.zeros((3,3))  # m_R_theta

    Masses = numpy.zeros((len(pchnodes),))
    Offsets = numpy.zeros((len(pchnodes),3))
    RawInertias = numpy.zeros((len(pchnodes),3,3))
    Inertias = numpy.zeros((len(pchnodes),3,3))
    CrossInertias = numpy.zeros((len(pchnodes),3,3))

    for iconm2,conm2 in enumerate(conm2s):
        totalMass += conm2.mass*numpy.eye(3)
        Masses[conm2onpchnode[iconm2]] += conm2.mass
        centroid = conm2.Centroid() # grid+offset in global coordinate systems
        offset = conm2.X # offset to grid in global coordinate system
        #print centroid
        inertia = conm2.Inertia() #Inertia about centroid global coordinate system
        D = skew(centroid)
        M = numpy.eye(3)*(conm2.mass)
        totalInertia += inertia - numpy.dot(M,numpy.dot(D,D)) # parallel axis theorm (wikipedia:moment of inertia)
        #print numpy.dot(M,numpy.dot(D,D))
        crossInertia += -numpy.dot(M,D)
        #print numpy.dot(M,D)
        lD = skew(offset) # local offset
        CrossInertias[conm2onpchnode[iconm2]] += -numpy.dot(M,lD)

        RawInertias[conm2onpchnode[iconm2]] = inertia
        Offsets[conm2onpchnode[iconm2]] = offset

        gridPos = pchnodes[conm2onpchnode[iconm2]].get_position()
        D = skew(centroid - gridPos)
        InertiaAboutGridPos = inertia - numpy.dot(M,numpy.dot(D,D))
        Inertias[conm2onpchnode[iconm2]] += InertiaAboutGridPos

    # for conm2,iNearest in lonelymatchedconm2s:
    #     totalMass += conm2.mass*numpy.eye(3)
    #     centroid = conm2.Centroid() # grid+offset in global coordinate systems
    #     #print centroid
    #     inertia = conm2.Inertia() #Inertia about centroid global coordinate system
    #     D = skew(centroid)
    #     M = numpy.eye(3)*(conm2.mass)
    #     totalInertia += inertia - numpy.dot(M,numpy.dot(D,D)) # parallel axis theorm (wikipedia:moment of inertia)
    #     #print numpy.dot(M,numpy.dot(D,D))
    #     crossInertia += -numpy.dot(M,D)
    #     #print numpy.dot(M,D)

    #     # Add to respective Masses
    #     print len(Masses),iNearest
    #     Masses[iNearest] += conm2.mass
    #     gridID = conm2.Nid()
    #     gridPos = matchedconm2s[iNearest].nid.get_position() # This is the position of the node to put this inertia on
    #     D = skew(centroid - gridPos)
    #     InertiaAboutGridPos = inertia - numpy.dot(M,numpy.dot(D,D))
    #     Inertias[iNearest] = InertiaAboutGridPos

    print
    print printmat(numpy.vstack(( numpy.hstack((totalMass,crossInertia)),
                                  numpy.hstack((crossInertia.T,totalInertia)))),"Total Inertia matrix" , "%14.6E")
    print

    return Masses,Inertias,CrossInertias,RawInertias,Offsets



def CalculateConstantInertiaShapeIntegrals(N_r,N_theta):
    ''' Calculation of constant inertia shape integrals: pg 229 Shabana
    N_r      [nnodes*3 x nmodes] : Mode shapes associated with translations
    N_theta  [nnodes*3 x nmodes] : Mode shapes associated with rotations
    u_bar_0      [nnodes x 3]    : coordinates of nodes with mass and inertia
    Masses   [nnodes x 1]        : Mass at each node
    Inertias [nnodes x 3 x 3]    : 3x3 inertia matricies of each node (about that node and in body coordinates)
    '''
    pass

def BuildNumberRangesFromStr(fullrangestr):

    rangestrs = fullrangestr.lstrip().rstrip().split(",")
    numrange_int = []
    for rangestr in rangestrs:
        if rangestr.find("-") >= 0 or  rangestr.find(":") >= 0:
            rangestr = rangestr.replace("-",",").replace(":",",")
            rangelist = eval("range(" + rangestr + ")")
            rangelist.append(rangelist[-1]+1)
            numrange_int.extend(rangelist)
        else:
            numrange_int.append(int(rangestr))

    #print numranges_int
    # sort and remove duplicates (?)
    numranges_sorted = sorted(list(set(numrange_int)))
    return numranges_sorted



#=======================================================================================================================================================


def run(bdfname,op2name,pchname,clampedRefNodes,op4,modesmaskstr,outputSuffix,replaceRBmodes):

    enforce = False

    if outputSuffix is not None:
        outputSuffix = "_" + outputSuffix
    else:
        outputSuffix = ""


    fem = BDF(debug=True,log=None)

    # Read bdf
    fem.cross_reference()
    fem.read_bdf(bdfname,xref=True)




    #fem.cross_reference()
    if not args.op4:
        # Read *.pch file
        print 'Reading pch file:',args.pch
        #id_list,E,A = read_pch(args.pch)
        id_list,stiffnessMatrix,massMatrix = read_pch(pchname)
    else:
        print "Reading op4 file",args.pch
        op4 = OP4()
        op4 = op4.read_op4(args.pch)
        (formE,stiffnessMatrix) = op4["KAA"]
        (formA,massMatrix) = op4["MAA"]
        print fem.asets[0].IDs
        id_list = zip(range(len(fem.asets[0].IDs)),fem.asets[0].IDs)
        #print id_list,E.shape
    print "len(id_list)",len(id_list)
    # use id_list to get conm2s in order
    conm2onpchnode,conm2s,pchnodes = LineUpCONM2s(fem,id_list)

    # Build inertia tensor and mass matrix from conm2s
    Masses,Inertias,CrossInertias,RawInertias,Offsets = ExtractCONM2Data(fem,conm2s,pchnodes,conm2onpchnode)

    # Build Mass matrix
    nnodes = len(id_list)
    MssCalc = numpy.zeros((nnodes*6,nnodes*6))

    for inode in range(nnodes):
        mij = Masses[inode]
        if mij < 1e-3:
            print "Tiny Mass: %f @ node %i" % (mij,inode)

        Iij = Inertias[inode]
        for ii in range(3):
            if Iij[ii,ii] < 1e-3:
                print "Tiny Inertia[%i,%i]: %f @ node %i" % (ii,ii,Iij[ii,ii],inode)

        MssCalc[inode*6:inode*6+3,inode*6:inode*6+3] += mij*numpy.eye(3)
        MssCalc[inode*6:inode*6+3,inode*6+3:inode*6+6] += CrossInertias[inode]
        MssCalc[inode*6+3:inode*6+6,inode*6:inode*6+3] += CrossInertias[inode].T
        MssCalc[inode*6+3:inode*6+6,inode*6+3:inode*6+6] += Iij



    massDiag = numpy.diag(MssCalc)
    print printmat(massDiag,"Diagonal of calculated mass matrix")



    # print "Test mass and stiffness matricies are symmetric"
    # print printmat(MssCalc.T - MssCalc,"Mass","%5.2g")
    # print printmat(stiffnessMatrix.T - stiffnessMatrix,"Stiffness","%5.2g")
    stiffnessMatrixReduced = stiffnessMatrix.copy()

    calculate=0
    if clampedRefNodes is not None:
        # Generate clamped mode shapes - the reference node is clamped.
        MssCalcReduced = MssCalc.copy()
        massMatrixReduced = massMatrix.copy()

        # Delete the rows and columns associated with the reference node from the mass and stiffness matrices
        clampedRefNodes = sorted(clampedRefNodes)
        for i,clampedRefNode in enumerate(clampedRefNodes):
            inds = slice((clampedRefNode-i)*6,(clampedRefNode-i+1)*6)
            stiffnessMatrixReduced = numpy.delete(stiffnessMatrixReduced,inds,0) # rows
            stiffnessMatrixReduced = numpy.delete(stiffnessMatrixReduced,inds,1) # columns
            MssCalcReduced = numpy.delete(MssCalcReduced,inds,0) # rows
            MssCalcReduced = numpy.delete(MssCalcReduced,inds,1) # columns
            massMatrixReduced = numpy.delete(massMatrixReduced,inds,0) # rows
            massMatrixReduced = numpy.delete(massMatrixReduced,inds,1) # columns

        write_structuralMatrix_file("massMatrixCalcRed%s" % outputSuffix,MssCalcReduced,"Reduced Mass Matrix","mass")

        # Calculate Eigenvectors and eigenvalues

        if calculate:
          eigval,eigvec = scipy.linalg.eigh(stiffnessMatrixReduced,MssCalcReduced)

        # Add zeros into eigenvectors.
        nmodes = (nnodes-len(clampedRefNodes))*6 # Removed nodes (6 less modes per node)
        print eigvec.shape,MssCalc.shape
        for i,clampedRefNode in enumerate(clampedRefNodes):
            j = len(clampedRefNodes)-i # counts down
            #print eigvec[0:clampedRefNode*6,:].shape, numpy.zeros((6,nmodes)).shape, eigvec[clampedRefNode*6:,:].shape
            eigvec = numpy.vstack((eigvec[0:(clampedRefNode-j)*6,:],numpy.zeros((6,nmodes)),eigvec[(clampedRefNode-j)*6:,:]))


    else:

        # print printmat(massMatrix,"MassMatrix",tol=1e-3)
        # print printmat(MssCalc,"MassMatrixCalc",tol=1e-10)
        if calculate:
         numpy.linalg.cholesky(MssCalc[0:5,0:5])
         # Calculate Eigenvectors and eigenvalues
         eigval,eigvec = scipy.linalg.eigh(stiffnessMatrix,MssCalc)
         #eigval,eigvec = scipy.linalg.eigh(stiffnessMatrix,massMatrix)


    if op2name is not None:
        op2 = OP2(debug=True,log=None)
        op2.read_op2(op2name)
        nodeids = [x[1] for x in id_list]

        eigvec,eigval = GetEigenvectors(fem,op2,nodeids,None)

        nmodes = eigvec.shape[0]
        nnodes = len(nodeids)

        phi = numpy.zeros((nnodes*6,nmodes))
        for inode in range(nnodes):
            for jmode in range(nmodes):
                phi[inode*6:(inode+1)*6,jmode] = eigvec[jmode,inode,:]


        eigvec = phi

        print "eigvec.shape",eigvec.shape


    if enforce:
        eigvalnewsort,eigvecnewsort = EnforceZeroEigvals(nnodes,eigvalsort,eigvecsort)
        eigval = eigvalnewsort.copy()
        eigvec = eigvecnewsort.copy()

    if calculate:
        # Test the eigenvectors
        modalMass = mdot(eigvec.T,MssCalc,eigvec)
        #print printmat(modalMass,"modalMass","%12.5g")

        modalStiffness = mdot(eigvec.T,stiffnessMatrix,eigvec)
        #print printmat(modalStiffness,"modalStiffness","%12.5g")

        # SAVE EVERYTHING TO FILE

        # Save eigenvalues to file
        freqs = numpy.sqrt(numpy.abs(numpy.real(eigval)))/(2.0*numpy.pi)
        sortinds = numpy.argsort(eigval,axis=0,kind='mergesort')

    #print printmat(eigval.reshape(len(eigval),1),"Eigenvalues","%12.5g")

    outputeigenvalues=0
    if outputeigenvalues:

        fd = open("eigenvalues%s" % outputSuffix,'w')
        fd.write("%21s %21s %10s\n" % ("Eigenvalues","Frequencies","SortOrder") )
        for irow in range(len(eigval)):
            fd.write("%22.15E %22.15E %10i\n" % (eigval[irow],freqs[irow],sortinds[irow]))
        fd.close()


        phi = eigvec.T.copy()

    #print numpy.shape(phi)
        nmodes = len(phi[:,0])
    if calculate:
        if modesmaskstr is not None:
            mask = numpy.ones(nmodes,dtype=bool)
            # Remove rigid body modes
            modes2keep = BuildNumberRangesFromStr(modesmaskstr)
            for i in range(len(mask)):
                if i not in modes2keep:
                    mask[sortinds[i]] = False

            phi = phi[mask]


        nmodes = len(phi[:,0])
        nnodes = len(phi[0,:])/6


    totalMass = numpy.sum(Masses)
    cg = numpy.sum(numpy.hstack([Masses[i]*(pchnodes[i].get_position().reshape((3,1))+Offsets[i].reshape((3,1))) for i in range(nnodes)]),axis=1)/totalMass
    if replaceRBmodes:


        print "cg",cg

        ri = numpy.zeros((nnodes*3,1))
        for i in range(nnodes):
            ri[i*3:i*3+3] = pchnodes[i].get_position().reshape((3,1)) - cg.reshape((3,1))

        phiR = numpy.zeros((6,6*nnodes))
        for i in range(nnodes):
            phiR[:,i*6:i*6+6] = numpy.vstack( ( numpy.hstack((numpy.eye(3), numpy.zeros((3,3)))),
                                                numpy.hstack((skew(ri[i*3:i*3+3]), numpy.eye(3))) ))

        # print phiR.shape,phi.shape
        phi[0:6,:] = phiR

    if calculate:
        print "modes",nmodes,"nodes",nnodes
        modalMass = mdot(phi,MssCalc,phi.T)
        print printmat(modalMass,"modalMass","%12.5g",1e-5)

    # Write to file
    structgrid = numpy.zeros((nnodes,4))
    inode = 0
    for inode,node in enumerate(pchnodes):
        structgrid[inode][0:3] = node.get_position()
        structgrid[inode][3] = node.nid-1

    #print structgrid

    numpy.save('CG',cg)
    numpy.save('Maa',massMatrix)
    numpy.save('Maac',MssCalc)
    numpy.save('Kaa',stiffnessMatrix)

    write_structuralGrid_file("structuralGrid",structgrid)
    #write_structuralMatrix_file("massMatrix%s" % outputSuffix,massMatrix,"Reduced Mass Matrix","mass")
    #write_structuralMatrix_file("massMatrixCalc%s" % outputSuffix,MssCalc,"Reduced Mass Matrix","mass")
    #write_structuralMatrix_file("stiffnessMatrix",stiffnessMatrix,"Reduced Stiffness Matrix","stiffness")
    #write_structuralModes_file("structuralModes%s" % outputSuffix,phi,nnodes,nmodes,"structural modes (eigenvectors) [nmodes,nnodes*6] nmodes=%i, nnodes=%i" % (nmodes,nnodes),"structuralmodes")
    #write_structuralVars_file("structuralVars",nnodes)
    #numpy.savez('structuralMassData%s.npz' % outputSuffix,Masses=Masses, Inertias=Inertias,CrossInertias=CrossInertias, RawInertias=Inertias, Offsets=Offsets)



    # Generate the inverted stiffness matrix (flexibility matrix)

    #print "Stiffness matrix condition number (small is good):",numpy.linalg.cond(stiffnessMatrixReduced)
    #invStiffnessMatrixReduced = None
    #if  numpy.linalg.cond(stiffnessMatrixReduced) < 1.0/sys.float_info.epsilon:
    #    invStiffnessMatrixReduced = numpy.linalg.inv(stiffnessMatrixReduced)
    #else:
       # raise ValueError("Stiffness Matrix is singular (It probably does not have enough constraints.)")
        #print "WARNING!!: Stiffness Matrix is singular (It probably does not have enough constraints.)"

    #if invStiffnessMatrixReduced is not None:
        #write_structuralMatrix_file("structuralMatrix",invStiffnessMatrixReduced,"Reduced Flexibility Matrix","flexibility")


    # k = nnodes*6 - 1
    # eigval,eigvec = scipy.sparse.linalg.eigsh(stiffnessMatrix,k,MssCalc,which="SM")

    #eigval,eigvec = scipy.linalg.eigh(mdot(numpy.linalg.inv(MssCalc),stiffnessMatrix))

    # print "#######################################################"
    # print "MATRICIES STRAIGHT AFTER EIG CALC BEFORE ANY FUNNY BUSINESS AT ALL"

    # # Check on matricies.
    # M = mdot(eigvec.T,MssCalc,eigvec)
    # print printmat(M,"modalMass","%12.5g",1e-10)

    # K = mdot(eigvec.T,stiffnessMatrix,eigvec)
    # print printmat(K,"modalStiffness","%12.5g",1e-10)


    # # Test eigenvectors are orthogonal
    # print "Test eigenvectors are orthogonal - before normalisation"
    # test = mdot(eigvec,eigvec.T)
    # print printmat(test,"eigvec orth test (should be identity matrix)","%12.5g",1e-10)

    # # Normalise eigenvectors to a norm of one
    # eigvecnorm = numpy.zeros_like(eigvec)
    # for iev in range(eigvec.shape[1]):
    #     eigvecnorm[:,iev] = eigvec[:,iev]/numpy.linalg.norm(eigvec[:,iev])

    # print "Test eigenvectors are orthogonal - after normalisation to one"
    # test = mdot(eigvecnorm,eigvecnorm.T)
    # print printmat(test,"eigvec orth test (should be identity matrix)","%12.5g",1e-10)



    # Mass normalisation of eigenvectors - This is done by eigh
    # M = mdot(eigvec.T,MssCalc,eigvec)
    # eigvec = numpy.sqrt(numpy.diag(numpy.linalg.inv(M)))*eigvec

    # # Test eigenvectors are orthogonal
    # print "Test eigenvectors are orthogonal - after mass normalisation"
    # test = mdot(eigvec,eigvec.T)
    # print printmat(test,"eigvec orth test (should be identity matrix)","%12.5g",1e-10)


    # print "#######################################################"
    # print "MATRICIES AFTER NORMALISATON "

    # # Check on matricies.
    # M = mdot(eigvec.T,MssCalc,eigvec)
    # print printmat(M,"modalMass","%12.5g",1e-10)

    # K = mdot(eigvec.T,stiffnessMatrix,eigvec)
    # print printmat(K,"modalStiffness","%12.5g",1e-10)

##### COMMENT FROM HERE

    # print "#######################################################"
    # print "SORTED EIGENVALUES AND VECTORS"

    # # Sort eigenvals - these are normalised so Mm = I
    # eigvalsort = numpy.sort(eigval,kind='mergesort')

    # print printmat(eigvalsort.reshape(len(eigvalsort),1),"Sorted Eigenvalues","%12.5g")

    # # sort eigenvectors
    # sortinds = numpy.argsort(eigval,kind='mergesort')
    # eigvecsort = numpy.take(eigvec,sortinds,axis=1)

    # # Check I got the take command right
    # for ind in range(10):
    #     print printmat(eigvecsort[:,ind].reshape((nnodes,6)),"Eigenvector No %i - Eigenvalue: %f" % (ind+1,eigval[sortinds[ind]]),"%12.5g")

    # print "########################################################"
    # print "MATRICIES AFTER SORTING"

    # # Check on matricies.
    # M = mdot(eigvecsort.T,MssCalc,eigvecsort)
    # print printmat(M,"modalMass","%12.5g",1e-10)

    # K = mdot(eigvecsort.T,stiffnessMatrix,eigvecsort)
    # print printmat(K,"modalStiffness","%12.5g",1e-10)

    # print "Test eigenvectors are orthogonal - after sorting"
    # test = mdot(eigvecsort,eigvecsort.T)
    # print printmat(test,"eigvec orth test (should be identity matrix)","%12.5g",1e-10)

    # print "#######################################################"
##### COMMENT TO HERE

    # Code to extract modes from NASTRAN


        # # Decide which modes to include -- default all
        # mask = numpy.ones(len(phi),dtype=bool)
        # # Remove rigid body modes
        # modes2keep = range(6,len(mask))
        # for i in range(len(mask)):
        #     if i not in modes2keep:
        #         mask[i] = False

        # phi = phi[mask]
        #eigvec = phi.T.copy()


    # Remove rigid body modes
    #print eigvec.shape

    # if numpy.any(eigval < 0.0):
    #     print "WARNING: NEGATIVE EIGENVALUES FOUND"
    # sortinds = numpy.argsort(eigval,kind='mergesort')
    # print printmat(numpy.hstack((eigval,sortinds)),"eigenvalues","%12.5g")
    # print printmat(numpy.sort(eigval),"Sorted eigenvalues","%12.5g")

    # print sortinds[0:6]
    # print [eigval[i] for i in sortinds[0:6]]

    # for ind in range(10):
    #     print printmat(eigvec[:,sortinds[ind]].reshape((nnodes,6)),"Eigenvector No %i - Eigenvalue: %f" % (ind+1,eigval[sortinds[ind]]),"%12.5g")


    # # eigvec = numpy.delete(eigvec,sortinds[0:6],1)
    # # print eigval.shape,eigvec.shape
    # # eigvalnew = numpy.delete(eigval.reshape((eigval.shape[0],1)),sortinds[0:6],0)
    # # print eigval.shape
    # eigvalnew = eigval.reshape((len(eigval),1))


    # fd.close()
    # If you want to delete more high frequency modes do this
    # eigvec = numpy.delete(eigvec,sortinds[-1],1) - deletes the mode with the highest freq

    # Do the enforcement

def readOP4_matrices(readKM=['K.op4','M.op4'],nameKM=['KAA','MAA'],saveKM=['/Kaa.npy','/Maa.npy']):
    model= OP4()
    numMat = len(readKM)
    Mat = [[] for j in range(numMat)]
    for i in range(numMat):
        Km = model.read_op4(readKM[i],precision='double')
        #pdb.set_trace()
        #Mm = model.read_op4(readKM[1])
        (formK, K) = Km[nameKM[i]]
        Mat[i] = K
        #(formM, M) = Mm[nameKM[1]]
    for si in range(len(saveKM)):
        numpy.save(saveKM[si],Mat[si])
    return Mat

def mat_componentsold(M,aset,asetrb={}):
    nM=numpy.shape(M)[0]
    add_nM = 0
    for i in asetrb.keys():
        add_nM +=len(asetrb[i])
    nM+=add_nM
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    jrb=0
    irb=0
    for j in range(nM):
        jn=(j)/6;jr=(j)%6
        if aset[jn] in asetrb.keys(): # Clamped or multibody node
            if jr in asetrb[aset[jn]]: # Removed DoF
                if jr == asetrb[aset[jn]][-1]: # Last DoF
                    jrb +=len(asetrb[aset[jn]]) # Add missing DoF to size of matrix
                pass
        irb=0
        for i in range(nM):
            #pdb.set_trace()
            inn=(i)/6;ir=(i)%6
            if aset[inn] in asetrb.keys(): # Clamped or multibody node
                if ir in asetrb[aset[inn]]: # Removed DoF
                    if ir == asetrb[aset[inn]][-1]: # Last DoF
                        irb +=len(asetrb[aset[inn]]) # Add missing DoF to size of matrix
                    pass

            if i<(j+1) and M[i-irb,j-jrb]!=0:
                GCj.append([aset[jn],jr+1])
                GCi.append([aset[inn],ir+1])
                GM.append(numpy.double(M[i-irb,j-jrb]))
    return GCi,GCj,GM

def mat_components(M,aset,asetrb={}):
    nM=numpy.shape(M)[0]
    add_nM = 0
    for i in asetrb.keys():
        add_nM +=len(asetrb[i])
    nM+=add_nM
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    jrb=0
    irb=0
    #pdb.set_trace()
    for j in range(nM):
        jn=(j)/6;jr=(j)%6
        if aset[jn] in asetrb.keys(): # Clamped or multibody node
            if jr in asetrb[aset[jn]]: # Removed DoF
                jrb += 1
                continue
        irb=0
        for i in range(nM):
            #pdb.set_trace()
            inn=(i)/6;ir=(i)%6
            if aset[inn] in asetrb.keys(): # Clamped or multibody node
                if ir in asetrb[aset[inn]]: # Removed DoF
                    irb +=1# Add missing DoF to size of matrix
                    continue

            if i<(j+1) and M[i-irb,j-jrb]!=0:
                GCj.append([aset[jn],jr+1])
                GCi.append([aset[inn],ir+1])
                GM.append(numpy.double(M[i-irb,j-jrb]))
    return GCi,GCj,GM

def mat_components2(M,aset,asetrb={}):
    nM=numpy.shape(M)[0]
    add_nM = 0
    for i in asetrb.keys():
        add_nM +=len(asetrb[i])
    nM+=add_nM
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    jrb=0
    irb=0
    #pdb.set_trace()
    for j in range(nM):
        jn=(j)/6;jr=(j)%6
        if aset[jn] in asetrb.keys(): # Clamped or multibody node
            if jr in asetrb[aset[jn]]: # Removed DoF
                jrb += 1
                continue
        irb=0
        for i in range(nM):
            #pdb.set_trace()
            inn=(i)/6;ir=(i)%6
            if aset[inn] in asetrb.keys(): # Clamped or multibody node
                if ir in asetrb[aset[inn]]: # Removed DoF
                    irb +=1# Add missing DoF to size of matrix
                    continue

            if i<(j+1) and M[i-irb,j-jrb]!=0:
                if aset[jn] in asetrb.keys():
                    if jr not in asetrb[aset[jn]]:
                        jrx=list(set(range(6))-set(asetrb[aset[jn]])).index(jr)
                    GCj.append([aset[jn],jrx+1])
                else:
                    GCj.append([aset[jn],jr+1])
                if aset[inn] in asetrb.keys():
                    if ir not in asetrb[aset[inn]]:
                        irx=list(set(range(6))-set(asetrb[aset[inn]])).index(ir)
                    GCi.append([aset[inn],irx+1])
                else:
                    GCi.append([aset[inn],ir+1])
                GM.append(numpy.double(M[i-irb,j-jrb]))
    return GCi,GCj,GM

def mat_components3(M,aset,asetrb={}):
    nM=numpy.shape(M)[0]
    add_nM = 0
    for i in asetrb.keys():
        add_nM +=len(asetrb[i])
    nM+=add_nM
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    jrb=0
    irb=0
    #pdb.set_trace()
    for j in range(nM):
        jn=(j)/6;jr=(j)%6
        if aset[jn] in asetrb.keys(): # Clamped or multibody node
            if jr in asetrb[aset[jn]]: # Removed DoF
                jrb += 1
                continue
        irb=0
        for i in range(nM):
            #pdb.set_trace()
            inn=(i)/6;ir=(i)%6
            if aset[inn] in asetrb.keys(): # Clamped or multibody node
                if ir in asetrb[aset[inn]]: # Removed DoF
                    irb +=1# Add missing DoF to size of matrix
                    continue

            if i<(j+1) and M[i-irb,j-jrb]!=0:
                GCj.append([aset[jn],jr+1])
                GCi.append([aset[inn],ir+1])
                GM.append(numpy.double(M[i-irb,j-jrb]))
    return GCi,GCj,GM


def mat_nonzero(k):
    count =0
    ni,nj = numpy.shape(k)
    for i in range(ni):
        for j in range(nj):
            if k[i,j] !=0 and j<=i:
                count+=1
    return count
def check_components(M,aset):
    nM=numpy.shape(M)[0]
    assert len(aset)*6==nM
    GCj=[]
    GCi=[]
    GM=[]
    for j in range(nM):
        jn=j/6;jr=j%6
        for i in range(j+1):
            if M[i,j]!=0:
                inn=i/6;irr=i%6
                GCj.append([aset[jn],jr+1])
                GCi.append([aset[inn],irr+1])
                GM.append(numpy.double(M[i,j]))
    return GCi,GCj,GM


def mat2dmig(aset,Mname,Mread=['KAA','MAA'],Msave=['KAAX','MAAX'],write_dmig='dmig.bdf',aset_coord=[],asetrb={}):
    model_dmig=BDF()
    if type(Mname).__name__ == 'str':
        Mname = [Mname]
    if type(Msave).__name__ == 'str':
        Msave = [Msave]
    extension = Mname[0].split('.')[-1]
    if extension == 'op4':
       Ka = readOP4_matrices(Mname,Mread,[])
    elif extension == 'npy':
      Ka = []
      for Mi in Mname:
          Ka.append(numpy.load(Mi))
    numKa=len(Ka)

    #Utils.FEM_MatrixBuilder.write_fem(fname='dmig2.txt',Mname='KAAX',M=Ka,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem(fname='dmig2.txt',Mname='MAAX',M=Ma,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem2(fname='dmig4.txt',Mname='KAAX',M=Ka,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem2(fname='dmig4.txt',Mname='MAAX',M=Ma,aset=aset)
    for i in range(numKa):
        #Utils.FEM_MatrixBuilder.write_fem3(Mname='KAAX',M=Ka,aset=aset)
        #Utils.FEM_MatrixBuilder.write_fem3(Mname='MAAX',M=Ma,aset=aset)
        KAAX_GCi,KAAX_GCj,KAAX_GM = mat_components(M=Ka[i],aset=aset,asetrb=asetrb)
        #MAAX_GCi,MAAX_GCj,MAAX_GM = write_fem3(M=Ma,aset=aset)
        model_dmig.add_dmig(Msave[i], 6, 2, 2,0, numpy.shape(Ka)[0],KAAX_GCj,KAAX_GCi,KAAX_GM, Complex=None, comment='Export DMIG Matrix')
        #model_dmig.add_dmig('MAAX', 6, 2, 2,0, numpy.shape(Ma)[0],MAAX_GCj,MAAX_GCi,MAAX_GM, Complex=None, comment='Export DMIG Matrix')
    if len(aset_coord)>0:
        for ai in range(len(aset)):
            if type(aset_coord).__name__ == 'dict':
                model_dmig.add_grid(aset[ai],aset_coord[aset[ai]])
            else:
                model_dmig.add_grid(aset[ai],aset_coord[ai])
    model_dmig.write_bdf(write_dmig,size=16,is_double=True)
    check_dmig = BDF()
    check_dmig.read_bdf(write_dmig)
    for i in range(numKa):
        Kax,del1,del2=check_dmig.dmigs[Msave[i]].get_matrix()
        #print Kax
        #print '######'
        #print Ka[i]
        assert numpy.allclose(Kax,Ka[i]), 'DMIG not well written'


def mat2dmig3(aset,Mname,Mread=['KAA','MAA'],Msave=['KAAX','MAAX'],write_dmig='dmig.bdf',aset_coord=[],asetrb={}):
    model_dmig=BDF()
    if type(Mname).__name__ == 'str':
        Mname = [Mname]
    if type(Msave).__name__ == 'str':
        Msave = [Msave]
    extension = Mname[0].split('.')[-1]
    if extension == 'op4':
       Ka = readOP4_matrices(Mname,Mread,[])
    elif extension == 'npy':
      Ka = []
      for Mi in Mname:
          Ka.append(numpy.load(Mi))
    numKa=len(Ka)

    #Utils.FEM_MatrixBuilder.write_fem(fname='dmig2.txt',Mname='KAAX',M=Ka,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem(fname='dmig2.txt',Mname='MAAX',M=Ma,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem2(fname='dmig4.txt',Mname='KAAX',M=Ka,aset=aset)
    #Utils.FEM_MatrixBuilder.write_fem2(fname='dmig4.txt',Mname='MAAX',M=Ma,aset=aset)
    for i in range(numKa):
        #Utils.FEM_MatrixBuilder.write_fem3(Mname='KAAX',M=Ka,aset=aset)
        #Utils.FEM_MatrixBuilder.write_fem3(Mname='MAAX',M=Ma,aset=aset)
        KAAX_GCi,KAAX_GCj,KAAX_GM = mat_components3(M=Ka[i],aset=aset,asetrb=asetrb)
        #MAAX_GCi,MAAX_GCj,MAAX_GM = write_fem3(M=Ma,aset=aset)
        model_dmig.add_dmig(Msave[i], 6, 2, 2,0, numpy.shape(Ka)[0],KAAX_GCj,KAAX_GCi,KAAX_GM, Complex=None, comment='Export DMIG Matrix')
        #model_dmig.add_dmig('MAAX', 6, 2, 2,0, numpy.shape(Ma)[0],MAAX_GCj,MAAX_GCi,MAAX_GM, Complex=None, comment='Export DMIG Matrix')
    if len(aset_coord)>0:
        for ai in range(len(aset)):
            if type(aset_coord).__name__ == 'dict':
                model_dmig.add_grid(aset[ai],aset_coord[aset[ai]])
            else:
                model_dmig.add_grid(aset[ai],aset_coord[ai])
    model_dmig.write_bdf(write_dmig,size=16,is_double=True)
    check_dmig = BDF()
    check_dmig.read_bdf(write_dmig)
    for i in range(numKa):
        Kax,del1,del2=check_dmig.dmigs[Msave[i]].get_matrix()
        #print Kax
        #print '######'
        #print Ka[i]
        assert numpy.allclose(Kax,Ka[i]), 'DMIG not well written'

        
#=====================================================================================================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts modes and various FEM data to build inertia matricies')
    parser.add_argument('bdf',type=str, help='bdf filename')
    parser.add_argument('pch',type=str, help='pch filename')
    parser.add_argument('--op2',type=str, default=None, help='op2 filename')
    parser.add_argument('--op4',action='store_true',default=False, help='The pch file is in nastran op4 format')
    parser.add_argument('-c','--clampedRefNode',type=int,nargs="+",default=None, help='Reference node to generate clamped mode shapes' )
    parser.add_argument('--modesmask',type=str, default=None, help='Modes mask e.g. 0-50')
    parser.add_argument('-o','--output',type=str, default=None, help='output file suffix - typically mass case designation')
    parser.add_argument('--replaceRBmodes',action='store_true',default=False, help='Replace Rigidbody modes from nastran with computed ones.')

    (args) = parser.parse_args()
    run(args.bdf,args.op2,args.pch,args.clampedRefNode,args.op4,args.modesmask,args.output,args.replaceRBmodes)
