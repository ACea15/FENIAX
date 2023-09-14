from pyNastran.bdf.bdf import BDF

class GenDLMPanels:

    def __init__(self,
                 components,
                 num_surfaces,
                 p1,
                 x12,
                 p4,
                 x43,
                 nspan,
                 nchord,
                 set1x,
                 spline_type=6):

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
