from pyNastran.bdf.bdf import read_bdf,BDF
import numpy as np

def convert_followingz(dbfname_in,bdfname_out,loading_gids,load_x,load_y,nstep):
  bdfmodel=read_bdf(dbfname_in,debug=None)
  bdfmodel.sol=400
  bdfmodel.case_control_deck.create_new_subcase(1)
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'SUBTITLE=load1')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'ANALYSIS = NLTRAN ')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'DISP(PLOT)=1')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'NLSTEP = 1')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'DLOAD = 8')
  #bdfmodel.case_control_deck.add_parameter_to_global_subcase('METHOD=None')
  bdfmodel.case_control_deck.add_parameter_to_global_subcase('SPCFORCES=None')
  bdfmodel.params={}
  bdfmodel.add_param('LGDISP',1)
  bdfmodel.add_param('POST',-1)
  #bdfmodel.add_param('AUTOMSET','YES')
  arg=['NLSTEP',1,load_x[-1]]+[None]*6+['GENERAL']+[None]*7+['FIXED',nstep]
  bdfmodel.add_card(arg,'NLSTEP')
  nids=[]
  for nid in bdfmodel.nodes:
    nids.append(nid)
  for i,loading_gid in enumerate(loading_gids):
    coord=bdfmodel.nodes[loading_gid].xyz
    assert i+1 not in nids, f'node {i+1} already exists'
    bdfmodel.add_grid(i+1,coord+np.array([0,0,1.0]))
    bdfmodel.add_rbe2(i+1,loading_gid,'123456',[i+1])
    bdfmodel.add_force1(i+1,loading_gid,1.0,loading_gid,i+1)
  bdfmodel.add_load(len(loading_gids)+1,1.0,[1.0]*len(loading_gids),[i+1 for i in range(len(loading_gids))])
  bdfmodel.add_dload(8,1.,1.,1)
  bdfmodel.add_tload1(1,len(loading_gids)+1,10)
  bdfmodel.add_tabled1(10,load_x,load_y)
  bdfmodel.add_mdlprm({'OFFDEF':'LROFF'})
  bdfmodel.write_bdf(bdfname_out)
  #delete unnecessary lines
  f=open(bdfname_out,'r')
  lines=f.read().splitlines()
  f.close()
  lines
  flag_aset1=False
  for i,line in enumerate(lines):
    if line[:4]=='SPCF' or line[:6]=='METHOD':
      lines[i]='$'+line
      continue
    if line[:5]=='ASET1':
      flag_aset1=True
      lines[i]='$'+line
      continue
    if flag_aset1:
      if line[:8]!=' '*8:
        flag_aset1=False
        continue
      lines[i]='$'+line
  f=open(bdfname_out,'w')
  f.write('\n'.join(lines))
  f.close()


  
