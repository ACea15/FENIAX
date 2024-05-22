# Downloads/ParaView-5.10.1-MPI-Linux-Python3.9-x86_64/bin/pvpython merge.py
cd vtk2d/paraview/
ffmpeg -framerate 60 -pattern_type glob -i '*.png'  -c:v mpeg4 -qscale:v 1 SimoFFB2D.mp4
