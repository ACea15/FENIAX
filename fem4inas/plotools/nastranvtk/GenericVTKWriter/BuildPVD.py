import os, glob, argparse, re

from PVDWriter import WritePVD

def get_file_idx(fname):
    thematch = re.search(r"I(\d+)",fname)
    if thematch:
        return int(thematch.group(1))
    else:
        thematch2 = re.search(r"i=(\d+)",fname)
        if thematch2:
            return int(thematch2.group(1))
        else:
            print(fname)
            raise ValueError("Filename does not contain a sequence number")

def get_time_from_fname(fname):
    '''Function to get the time value from a file name '''
    thematch = re.search(r"t=(\d+.\d+[eE][+-]\d+)",fname)
    thematch2 = re.search(r"i=(\d+)",fname)
    if thematch:
        time = float(thematch.group(1))
    else:
        time = get_file_idx(fname)
    
    return time

def run(vtmdir):
    
    # 1. Find vtms
    vtmfiles = glob.glob(os.path.join(vtmdir,"*.vtm"))
    
    # 2. Sort and extract times/iters from file names
    vtmfiles = sorted(vtmfiles,key=lambda l:get_file_idx(l))

    times = [get_time_from_fname(x) for x in vtmfiles]
    
    # Write PVD
    filedata = zip(vtmfiles,times)
    
    ii = os.path.basename(vtmfiles[0]).find("I")
    if ii < 0:
        ii = os.path.basename(vtmfiles[0]).find("i=")
    
    pvdfilename = os.path.basename(vtmfiles[0])[0:ii-1] + ".pvd"
    
    print("Writing:",pvdfilename)
    
    WritePVD(pvdfilename,filedata)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build paraview data file (pvd) from a directory containing vtm files')
    parser.add_argument('vtmdir',nargs="*",type=str,default=[os.getcwd()],help='directory containing VTK multiblock files (.vtm)')

    (args) = parser.parse_args()

    run(args.vtmdir[0])
