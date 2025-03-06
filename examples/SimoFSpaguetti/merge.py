from paraview.simple import *
import pathlib

# # List of .vtu files to merge
# file_list = [
#     '/media/acea/work/projects/FEM4INAS/examples/wingSP/generate_dynamics/sp_610.vtk/CBUSH.vtu',
#     '/media/acea/work/projects/FEM4INAS/examples/wingSP/generate_dynamics/sp_610.vtk/CQUAD4.vtu',
#     #'/media/acea/work/projects/FEM4INAS/examples/wingSP/generate_dynamics/sp_0.vtk/RBE2.vtu',
#     '/media/acea/work/projects/FEM4INAS/examples/wingSP/generate_dynamics/sp_610.vtk/RBE3.vtu'
#     # Add more file names as needed
# ]

# # Create a reader for each file
# readers = [XMLUnstructuredGridReader(FileName=file) for file in file_list]

# # Append the readers to merge the datasets
# appended = AppendDatasets(Input=readers)

# # Save the merged dataset
# writer = XMLUnstructuredGridWriter(Input=appended, FileName='/media/acea/work/projects/FEM4INAS/examples/wingSP/generate_dynamics/sp_610.vtk/merged_output.vtu')
# writer.UpdatePipeline()

def merge_paraview(file_list, file_out):
    # Create a reader for each file
    readers = [XMLUnstructuredGridReader(FileName=file) for file in file_list]

    # Append the readers to merge the datasets
    appended = AppendDatasets(Input=readers)

    # Save the merged dataset
    writer = XMLUnstructuredGridWriter(Input=appended, FileName=file_out)
    writer.UpdatePipeline()

def find_seriesfiles(folder, pattern):
    ...


paraview_files = ['CQUAD4.vtu',
                  'CONM2.vtu',
                  #'CBUSH.vtu',
                  #'RBE2.vtu',
                  #'CTRIA3.vtu',
                  'RBE3.vtu']

folder = pathlib.Path('./vtk2d')
folder_out = folder / "merged"
folder_out.mkdir(exist_ok=True)
for i, fi in enumerate(folder.glob("conf_*")):
    if fi.is_dir() and (fi / paraview_files[0]).is_file():
        #print(fi)
        name_len = len("conf_")
        index = fi.name[name_len:].split('.')[0]
        file_list = [str(fi / pvf) for pvf in paraview_files]
        #print(file_list)
        file_out = str(folder_out / f"conf_{index}.vtu")
        # readers = [XMLUnstructuredGridReader(FileName=file) for file in file_list]

        # # Append the readers to merge the datasets
        # appended = AppendDatasets(Input=readers)
        # print(file_out)
        # # Save the merged dataset
        # writer = XMLUnstructuredGridWriter(Input=appended, FileName=file_out)
        # writer.UpdatePipeline()        
        merge_paraview(file_list, file_out)

        
        
