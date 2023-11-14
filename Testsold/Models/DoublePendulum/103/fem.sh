#!/bin/bash
python /media/pcloud/Computations/FEM4INAS/Utils/FEM_MatrixBuilder.py n1.bdf n1.pch
mv -t ../FEM/ Maa.npy Maac.npy Kaa.npy CG.npy structuralGrid
cd ../FEM/
python transform_mat.py 
