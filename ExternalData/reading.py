import pdb
import os
import csv
import numpy as np
import pickle


cwd = os.getcwd()
folder = 'hesse68'
inid = 'hesse68'
disx = ['-t1','-t2_5','-t4','-t6','-t8','-t10r']

folder = 'Simo45-dead'
inid=''
disx=['u1','u2','u3']

folder = 'Simo68'
inid=''
disx=['t1','t2_5','t4','t6','t8','t10']

files=os.listdir(cwd+'/'+folder)
dic={}
for ifile in files:
    if ifile.split('.')[1]=='csv':
        cx=[]
        cy=[]
        with open(cwd+'/'+folder+'/'+ifile,'rb') as csvfile:
            lis=csv.reader(csvfile)
            for i in lis:
                cx.append(eval(i[0]))
                cy.append(eval(i[1]))
            dic[ifile.split('.')[0]+'x']=cx
            dic[ifile.split('.')[0]+'y']=cy

discrete=[]
output=[]
for i in range(len(disx)):
    discrete.append([inid+disx[i]+'x',inid+disx[i]+'y'])
    output.append([dic[discrete[i][0]],dic[discrete[i][1]]])

with open(folder+'.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
