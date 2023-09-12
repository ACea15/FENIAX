import numpy as np
import pyvista
import pathlib

def line_points(n: int, p1: np.ndarray, p2:np.ndarray) -> np.ndarray:
    """Array with n+1 points from p1 to p2

    Parameters
    ----------
    n : int
        Number of points
    p1 : np.ndarray
        Point 1
    p2 : np.ndarray
        Point 2

    Returns
    -------
    np.ndarray
        (n+1, 3) array with location of points

    """

    if np.allclose(p1, p2):
        raise ValueError("p1 equals p2")
    pn = (p2-p1)/np.linalg.norm(p2-p1)
    step = np.linalg.norm(p2-p1)/n
    px=[]
    for i in range(n):
      px.append(p1+step*i*pn) 
    px.append(p2)
    px=np.array(px)
    return px

def line_points_y(n: int, p1: np.ndarray, p2:np.ndarray) -> np.ndarray:
    """Array with n+1 points from p1 to p2

    Parameters
    ----------
    n : int
        Number of points
    p1 : np.ndarray
        Point 1
    p2 : np.ndarray
        Point 2

    Returns
    -------
    np.ndarray
        (n+1, 3) array with location of points

    """

    if np.allclose(p1, p2):
        raise ValueError("p1 equals p2")
    pn = (p2-p1) / (p2[1] - p1[1])
    step = (p2[1]-p1[1]) / n
    px=[]
    for i in range(n):
      px.append(p1+step*i*pn) 
    px.append(p2)
    px=np.array(px)
    return px

def caero2corners(le1, le2, l1, l2):

    te1 = [le1[0] + l1, le1[1], le1[2]]
    te2 = [le2[0] + l2, le2[1], le2[2]]
    return np.vstack([le1, le2, te1, te2])

def lt_points(n_span, le1, le2, te1, te2):

    leading_edge = line_points(n_span, le1, le2)
    trailing_edge = line_points(n_span, te1, te2)
    return leading_edge, trailing_edge

def caero2grid(components, caeros):

    component_grid = {k: dict() for k in components}
    for i, k in enumerate(components):
        Xcorners = caero2corners(caeros[i]['p1'],
                                 caeros[i]['p4'],
                                 caeros[i]['x12'],
                                 caeros[i]['x43'])
        leading_edge, trailing_edge = lt_points(caeros[i]['nspan'],
                                                Xcorners[0],
                                                Xcorners[1],
                                                Xcorners[2],
                                                Xcorners[3])
        component_grid[k]['leading_edge'] = leading_edge
        component_grid[k]['trailing_edge'] = trailing_edge
        component_grid[k]['chordwise_points'] = caeros[i]['nchord']
    return component_grid

def build_pyvista(le, te, chordwise_points=10) -> tuple[np.ndarray, np.ndarray]:

    points = []
    cells = []
    len_i = len(le)
    len_j = chordwise_points + 1 
    for i in range(len_i - 1):
        line_ = line_points(chordwise_points, le[i], te[i])
        if i==0:
            points = line_
        else:
            points = np.vstack([points, line_])
        for j in range(chordwise_points - 1):
            cells.append([4, i*len_j + j, (i+1)*len_j + j, (i + 1)*len_j + j+1, i*len_j + j +1])
    line_ = line_points(chordwise_points, le[i+1], te[i+1])
    points = np.vstack([points, line_])        
    return points, np.array(cells)

def build_gridmesh(components, save_file=None,
                   save_dir=None):

    for k, v in components.items():
    
        _points, _cells = build_pyvista(v['leading_edge'],
                                        v['trailing_edge'],
                                        v['chordwise_points'])
        mesh = pyvista.PolyData(_points, _cells)
        if save_file is not None:
            if save_dir is not None:
                file_path = pathlib.Path(save_dir)
            else:
                file_path = pathlib.Path().cwd() / "paraview"
            file_path.mkdir(parents=True, exist_ok=True)
            mesh.save(file_path / f"{save_file}_{k}.ply",
                      binary=False)
