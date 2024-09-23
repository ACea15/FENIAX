from pyNastran.bdf.bdf import read_bdf
import numpy as np

def followingz_static(bdfname_in,bdfname_out,loading_gids,load):
  bdfmodel=read_bdf(bdfname_in,debug=None)
  bdfmodel.sol=101
  bdfmodel.case_control_deck.create_new_subcase(1)
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'SUBTITLE=load1')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,'DISP(PLOT)=1')
  bdfmodel.case_control_deck.add_parameter_to_local_subcase(1,f'LOAD = {len(loading_gids)+1}')
  bdfmodel.params={}
  bdfmodel.add_param('POST',-1)
  nids=[]
  for nid in bdfmodel.nodes:
    nids.append(nid)
  for i,loading_gid in enumerate(loading_gids):
    coord=bdfmodel.nodes[loading_gid].xyz
    assert i+1 not in nids, f'node {i+1} already exists'
    bdfmodel.add_grid(i+1,coord+np.array([0,0,1.0]))
    bdfmodel.add_rbe2(i+1,loading_gid,'123456',[i+1])
    bdfmodel.add_force1(i+1,loading_gid,1.0,loading_gid,i+1)
  bdfmodel.add_load(len(loading_gids)+1,1.0,[load]*len(loading_gids),[i+1 for i in range(len(loading_gids))])
  #bdfmodel.add_mdlprm({'OFFDEF':'LROFF'})
  bdfmodel.write_bdf(bdfname_out)
  #delete unnecessary lines
  f=open(bdfname_out,'r')
  lines=f.read().splitlines()
  f.close()
  flag_aset1=False
  for i,line in enumerate(lines):
    if line[:4]=='SPCF' or line[:6]=='METHOD' or line[:5]=='EIGRL':
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