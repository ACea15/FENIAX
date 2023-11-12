from pyNastran.bdf.bdf import BDF
from fem4inas.preprocessor.utils import dump_yaml
import pathlib
from ruamel.yaml import YAML

class GenDLMPanels:

    def __init__(self,
                 components: list,
                 num_surfaces: int,
                 p1: list,
                 x12: list ,
                 p4: list,
                 x43: list,
                 nspan: list,
                 nchord: list,
                 set1x: list,
                 spline_type: int =6):

        self.components = components
        self.num_surfaces = num_surfaces
        self.p1 = p1
        self.x12 = x12
        self.p4 = p4
        self.x43 = x43
        self.nspan = nspan
        self.nchord = nchord
        self.set1x = set1x
        self.spline_type = spline_type
        self.build_dlm()

    def __eq__(self, o):
        equal_dict = dict(
            components=self.components==o.components,
            num_surfaces=self.num_surfaces==o.num_surfaces,
            p1=self.p1==o.p1,
            x12=self.x12==o.x12,
            p4=self.p4==o.p4,          
            x43=self.x43==o.x43, 
            nspan=self.nspan==o.nspan,
            nchord=self.nchord==o.nchord,
            se1x=self.set1x==o.set1x,
            spline_type=self.spline_type==o.spline_type
            )
        equal = False if False in equal_dict.values() else True
        if not equal:
            print("The following items are not equal:")
            [print(k) for k, v in equal_dict.items() if v == False]
        return equal
    
    @classmethod
    def from_file(cls, file_dir: str|pathlib.Path, **kwargs):
        yaml = YAML()
        yaml_dict = yaml.load(pathlib.Path(file_dir))
        return cls(**yaml_dict)
    
    @staticmethod
    def dlm1(num_surfaces,paero1,caero1,aelist,set1,spline67,p1,x12,p4,x43,nspan,nchord,set1x, spline_type):
        npanels = 0
        for i in range(num_surfaces):
            paero1[i]['pid'] = 950+i
            caero1[i]['eid'] = 10000000 +i*10000
            caero1[i]['pid'] = paero1[0]['pid'] # PAERO1
            caero1[i]['igroup'] = 1  # Group number
            #caero1[i]['igid'] = 1  # Group number
            caero1[i]['p1'] = p1[i] #[6.383,2.793,-0.981]   # 1-4|2-3 parallelogram points
            caero1[i]['x12'] = x12[i] #28.531-6.383# Distance from 1 to 2
            caero1[i]['p4'] = p4[i]#[6.383,-2.793,-0.981]   #  "
            caero1[i]['x43'] = x43[i]#28.531-6.383  # "
            caero1[i]['nspan'] = nspan[i]#4  # boxes across y-direction
            caero1[i]['nchord'] = nchord[i]#10  # boxes across x-direction
            aelist[i]['sid'] = 666000 +i
            aelist[i]['elements'] = list(range(caero1[i]['eid'],caero1[i]['eid']+
                                          caero1[i]['nspan']*caero1[i]['nchord']))
            set1[i]['sid'] = aelist[i]['sid']
            set1[i]['ids'] = set1x[i]
            spline67[i]=['SPLINE%s'%spline_type, aelist[i]['sid'],caero1[i]['eid'],aelist[i]['sid'],None,
                             set1[i]['sid']]  # EID,CAERO,AELIST,NONE,SETG
            npanels +=  nspan[i]*nchord[i]
        return npanels


    def build_dlm(self):

        self.caero1 = [{} for i in range(self.num_surfaces)]
        self.paero1 = [{} for i in range(self.num_surfaces)]
        self.spline67 = [[] for i in range(self.num_surfaces)]
        self.aelist = [{} for i in range(self.num_surfaces)]
        self.set1 = [{} for i in range(self.num_surfaces)]
        self.npanels = self.dlm1(self.num_surfaces,
                                 self.paero1,
                                 self.caero1,
                                 self.aelist,
                                 self.set1,
                                 self.spline67,
                                 self.p1,
                                 self.x12,
                                 self.p4,
                                 self.x43,
                                 self.nspan,
                                 self.nchord,
                                 self.set1x,
                                 self.spline_type)

    def build_grid(self):
        ...
        
    def build_model(self, model=None):
        if model is None:
            self.model = BDF(debug=True,log=None)
        else:
            self.model = model
        for i in range(self.num_surfaces):
            self.model.add_aelist(**self.aelist[i], comment=self.components[i])
            self.model.add_set1(**self.set1[i], comment=self.components[i])
            self.model.add_caero1(**self.caero1[i], comment=self.components[i])
            self.model.add_paero1(**self.paero1[i], comment=self.components[i])
            self.model.add_card(self.spline67[i],
                                f'SPLINE{self.spline_type}',
                                comment=self.components[i])
    def save_yaml(self, file_name):
        """
        Saves to YAML file the inputs necessary to construct the object
        """
        dictout = dict(
            components = [self.components, "DLM component names"],
            num_surfaces = [self.num_surfaces, "DLM number of components"],
            p1 = [self.p1, "Leading-edge inwards point"],
            x12 = [self.x12, "Chord length at p1"],
            p4 = [self.p4, "Leading-edge outwards point"],
            x43 = [self.x43, "Chord length at point p4"],
            nspan = [self.nspan, "Number of panels spanwise"],
            nchord = [self.nchord, "Number of panels chordwise"],
            set1x = [self.set1x, "Structural ids associated with each component"],
            spline_type = [self.spline_type, "Nastran spline 6 or 7"]
        )
        dump_yaml(file_name, dictout)
        
class GenFlutter:

    # https://pynastran-git.readthedocs.io/en/latest/reference/bdf/cards/aero/pyNastran.bdf.cards.aero.dynamic_loads.html

    def __init__(self,
                 flutter_id,
                 density_fact,
                 mach_fact,
                 kv_fact,
                 machs,
                 reduced_freqs,
                 u_ref=1.,
                 c_ref=1.,
                 rho_ref=1.,
                 flutter_method="PK",
                 flutter_sett=None,
                 aero_sett=None):

        self.flutter_id = flutter_id
        self.density_fact = density_fact
        self.mach_fact = mach_fact
        self.kv_fact = kv_fact
        self.machs = machs
        self.reduced_freqs = reduced_freqs
        self.u_ref = u_ref
        self.c_ref = c_ref
        self.rho_ref = rho_ref
        self.flutter_method = flutter_method
        if flutter_sett is None:
            self.flutter_sett = {}
        else:
            self.flutter_sett = flutter_sett
        if aero_sett is None:
            self.aero_sett = {}
        else:
            self.aero_sett = aero_sett

    def build_model(self, model=None):
        
        if model is None:
            self.model=BDF(debug=True,log=None)
        else:
            self.model = model
            
        self.model.add_aero(velocity=self.u_ref,
                            cref=self.c_ref,
                            rho_ref=self.rho_ref)
        self.model.add_flfact(9011, self.density_fact,
                              comment="Density factors")
        self.model.add_flfact(9012, self.mach_fact,
                              comment="Density factors")
        self.model.add_flfact(9013, self.kv_fact,
                              comment="Reduced_freq. or velocity factors")
        self.model.add_flutter(
            sid=self.flutter_id,
            density=9011,
            mach=9012,
            method=self.flutter_method,
            reduced_freq_velocity=9013,
            **self.flutter_sett)
        self.model.add_mkaero1(self.machs,
                               self.reduced_freqs,
                               comment="sampled Mach numbers and reduced freqs.")

def dlm_control_nodes(aero_mesh,file_save=''):

    tipo=''
    i=0
    ngrid=0
    nelem=0

    aero_mesh_file = open(aero_mesh, 'r')

    for line in aero_mesh_file :
            i=i+1
    linee1=i
    aero_mesh_file.seek(0)
    for k in range(linee1):
        tipo=aero_mesh_file.readline(4)
        aero_mesh_file.readline()
        if tipo=='GRID' :
            ngrid=ngrid+1
        if tipo=='CQUA' :
            nelem=nelem+1

    Matrice_grid=zeros((ngrid,4))
    Matrice_elem=zeros((nelem,5))
    Control_node=zeros((nelem,3))
    aero_mesh_file.seek(0)

    counter_grid=0
    counter_elem=0

    for k in range(linee1):
        tipo=aero_mesh_file.readline(8)
        tipo=tipo.strip()
        if tipo=='GRID*' :
            Matrice_grid[counter_grid,0]=int(aero_mesh_file.readline(16))
            aero_mesh_file.readline(16)
            Matrice_grid[counter_grid,1]=float(aero_mesh_file.readline(16))
            Matrice_grid[counter_grid,2]=float(aero_mesh_file.readline())
            aero_mesh_file.readline(8)
            Matrice_grid[counter_grid,3]=float(aero_mesh_file.readline())        
            counter_grid=counter_grid+1
        if tipo=='GRID' :
            Matrice_grid[counter_grid,0]=int(aero_mesh_file.readline(8))
            aero_mesh_file.readline(8)
            Matrice_grid[counter_grid,1]=float(aero_mesh_file.readline(8))
            Matrice_grid[counter_grid,2]=float(aero_mesh_file.readline(8))
            Matrice_grid[counter_grid,3]=float(aero_mesh_file.readline())        
            counter_grid=counter_grid+1
        if tipo=='CQUAD4*' :
            Matrice_elem[counter_elem,0]=int(aero_mesh_file.readline(16))
            aero_mesh_file.readline(16)
            Matrice_elem[counter_elem,1]=int(aero_mesh_file.readline(16))
            Matrice_elem[counter_elem,2]=int(aero_mesh_file.readline())
            aero_mesh_file.readline(8)
            Matrice_elem[counter_elem,3]=int(aero_mesh_file.readline(16))    
            Matrice_elem[counter_elem,4]=int(aero_mesh_file.readline())
            counter_elem=counter_elem+1
        if tipo=='CQUAD4' :
            Matrice_elem[counter_elem,0]=int(aero_mesh_file.readline(8))
            aero_mesh_file.readline(8)
            Matrice_elem[counter_elem,1]=int(aero_mesh_file.readline(8))
            Matrice_elem[counter_elem,2]=int(aero_mesh_file.readline(8))
            Matrice_elem[counter_elem,3]=int(aero_mesh_file.readline(8))    
            Matrice_elem[counter_elem,4]=int(aero_mesh_file.readline())
            counter_elem=counter_elem+1
        if tipo!='CQUAD4*' and tipo!='GRID*' and tipo!='CQUAD4' and tipo!='GRID' :
            aero_mesh_file.readline()

    point_1=zeros((3))
    point_2=zeros((3))
    point_3=zeros((3))
    point_4=zeros((3))

    for k in range(nelem):
        for i in range(ngrid):
            if int(Matrice_elem[k,1]) == int(Matrice_grid[i,0]):
               point_1[0]=Matrice_grid[i,1]
               point_1[1]=Matrice_grid[i,2]
               point_1[2]=Matrice_grid[i,3]
            if int(Matrice_elem[k,2]) == int(Matrice_grid[i,0]):
               point_2[0]=Matrice_grid[i,1]
               point_2[1]=Matrice_grid[i,2]
               point_2[2]=Matrice_grid[i,3]
            if int(Matrice_elem[k,3]) == int(Matrice_grid[i,0]):
               point_3[0]=Matrice_grid[i,1]
               point_3[1]=Matrice_grid[i,2]
               point_3[2]=Matrice_grid[i,3]
            if int(Matrice_elem[k,4]) == int(Matrice_grid[i,0]):
               point_4[0]=Matrice_grid[i,1]
               point_4[1]=Matrice_grid[i,2]
               point_4[2]=Matrice_grid[i,3]

        Control_node[k,0]=(((point_2[0]-point_1[0])*3/4+point_1[0])+((point_3[0]-point_4[0])*3/4+point_4[0]))/2        
        Control_node[k,1]=(((point_4[1]-point_1[1])*1/2+point_1[1])+((point_3[1]-point_2[1])*1/2+point_2[1]))/2
        Control_node[k,2]=(((point_4[2]-point_1[2])*1/2+point_1[2])+((point_3[2]-point_2[2])*1/2+point_2[2]))/2  

    aero_mesh_file.close()              

    if file_save:
        Control_node_file= open(file_save, 'w')

        for k in range(nelem):
            Control_node_file.write(str(Control_node[k,0]).ljust(16))
            Control_node_file.write(str(Control_node[k,1]).ljust(16))
            Control_node_file.write(str(Control_node[k,2]).ljust(16))
            Control_node_file.write(str('\n'))

        Control_node_file.close()

    return Control_node
