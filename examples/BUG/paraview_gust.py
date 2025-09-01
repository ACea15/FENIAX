# [[file:modelgeneration.org::*Plot][Plot:3]]
from paraview.simple import *
import pathlib

def merge_paraview(file_list, file_out):
    # Create a reader for each file
    readers = [XMLUnstructuredGridReader(FileName=file) for file in file_list]

    # Append the readers to merge the datasets
    appended = AppendDatasets(Input=readers)

    # Save the merged dataset
    writer = XMLUnstructuredGridWriter(Input=appended, FileName=file_out)
    writer.UpdatePipeline()


paraview_files = ['CQUAD4.vtu',
                  'CONM2.vtu',
                  #'CBUSH.vtu',
                  'RBE2.vtu',
                  'CTRIA3.vtu',
                  'RBE3.vtu',
                  'CBAR.vtu']

folder = pathlib.Path("./results/gust200_eao/paraview") # pathlib.Path('./paraview/soldyn1')
folder_out = folder / "merged"
folder_out.mkdir(exist_ok=True, parents=True)
for i, fi in enumerate(folder.glob("bug_*")):
    if fi.is_dir() and (fi / paraview_files[0]).is_file():
        #print(fi)
        name_len = len("bug_")
        index = fi.name[name_len:].split('.')[0]
        file_list = [str(fi / pvf) for pvf in paraview_files]
        #print(file_list)
        file_out = str(folder_out / f"bug_{index}.vtu")
        # readers = [XMLUnstructuredGridReader(FileName=file) for file in file_list]

        # # Append the readers to merge the datasets
        # appended = AppendDatasets(Input=readers)
        # print(file_out)
        # # Save the merged dataset
        # writer = XMLUnstructuredGridWriter(Input=appended, FileName=file_out)
        # writer.UpdatePipeline()        
        merge_paraview(file_list, file_out)
# Plot:3 ends here
