# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:55:34 2022

@author: hangargs
"""
import os
import pickle
from time import time
import numpy as np
from stl import mesh
import numpy.linalg as LA
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


def read_process_mesh(filename, precision=0.001):
    """
    Reads .stl mesh file, extracts vertices, normals and triangles from mesh.
    Removes duplicated vertices, corresponding cormals and updates triangles array.
    And returns dictionary with processed vertices, triangles, columns.

    Parameters
    ----------
    filename : str
        Name or filepath of stl file.
    precision : float, optional
        Club points within radius of precision.

    Returns
    -------
    new : dict with keys ['N', 'V', 'T']
        All processed attributes are stored as dictionary.
        'V': deduplicated vertices of triangles,
        'N': removed normals corresponding to duplicate vertices,
        'T': updated triangles as per new vertices
    """
    #proceesed attributes will be stored in new dictionary
    new = {}

    #read stl file as mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    #extracting attributes from mesh file
    p_vert = np.asarray(mesh.vertices)
    p_norm = np.asarray(mesh.vertex_normals)
    p_tris = np.asarray(mesh.triangles).astype(np.int64)

    #removing duplicate points from vertices set
    grouped = deduplicate_points(p_vert, precision)

    #initializing empty arrays
    new['T'] = p_tris.copy()
    id_list = []

    #storing values of attributes
    for i, key in enumerate(grouped):
        np.copyto(new['T'], i, where=np.isin(p_tris, key))
        id_list.append(key[0])

    #removing invalid triangles from array
    new['V'] = p_vert[id_list]
    new['N'] = p_norm[id_list]
    new['T'] = filter_triangles(new)
 
    return new

def read_process_thck(filename, target, limit, dlm=","):
    """
    Reads .csv file, extracts target thickness and imputes thickness
    for missing points.

    Parameters
    ----------
    filename : str
        Name or filepath of stl file.
    target : ndarray
        Points from original mesh file.

    Returns
    -------
    y_true : nd_array
        true thickness associated with point in vertices array.
    """

    thk_data = np.loadtxt(filename, delimiter=dlm, dtype=float)
    tmp_thk = thk_data[(thk_data[:, 3]<limit) & (thk_data[:, 3]>0.05)]
    tmp_pnt = tmp_thk[:, :3]
    tmp_thk = tmp_thk[:, 3]

    dist, indx = cKDTree(tmp_pnt).query(target, k=3)
    
    y_true = np.where(
        dist[:, 0]<0.02, tmp_thk[indx[:, 0]], np.mean(tmp_thk[indx], axis=1)
    )

    return y_true

def filter_triangles(part_dict):
    """
    Returns triangles passing validity checks :
        1. Triangles should have non duplicate index
        2. Triangles should not have colinear vertices
        3. Triangle should not overlapp wth any other triangle

    Parameters
    ----------
    part_dict : dict,
        dictionary with vertex, normals, triangls of part.


    Returns
    -------
    tri_array : ndarray, shape-(no_of_triangles, 3), dtype-int
        2D array where each row is validated triangle with index of point
        in verex_array.
    """

    #initializing output array
    tri_array = part_dict['T'].copy()

    #mask for non-duplicate vertices
    mask = np.apply_along_axis(lambda t: len(set(t))==3, 1, tri_array)
    tri_array = tri_array[mask]

    #mask for non-overlapping triangles
    tmp_array = tri_array.copy()
    tmp_array.sort(axis=-1)
    _, mask = np.unique(tmp_array, return_index=True, axis=0)
    tri_array = tri_array[mask]

    #mask for non-linear vertices
    xyz_array = part_dict['V'][tri_array]
    s_ab = xyz_array[:, 1, :] - xyz_array[:, 0, :]
    s_bc = xyz_array[:, 2, :] - xyz_array[:, 1, :]
    mask = np.linalg.norm(np.cross(s_ab, s_bc), axis=-1)>(1e-5)
    tri_array = tri_array[mask]

    return tri_array

def deduplicate_points(points, radius):
    """
    Creates group of points which are within ball radius.

    Parameters
    ----------
    points : ndarray, shape (no_of_points, 3)
        2D array where each row is x, y, z co-ordinates of points.
    radius : float, scalar
        Radius of precision, points within the radius are considered to be same.

    Returns
    -------
    grouped : ndarray, shape (no_of_unique_clusters, dtype=list)
    """
    #building KDTree for efficient search
    point_tree = cKDTree(points)

    #searching points within precision radius
    grouped = point_tree.query_ball_point(points, radius)

    #deduplicating array of groups
    grouped = np.unique(grouped)

    return grouped

def get_grid_points(n_side=10, n_cell=32):
    """
    Creates uniform square cartesian grid in 2D plane with origin (0, 0),
    each side equal to *n_side* and no of cells equal to *n_cell*

    Array of x cordinates and y coordinates is returned.

    Parameters
    ----------
    n_side : float, scalar
        Length of any side of square, can be of any size.
    n_cell : int, scalar
        Number of cells in square grid, along one side.

    Returns
    -------
    x : ndarray, shape (n_cell, )
    y : ndarray, shape (n_cell, )
    """
    #extracting parameters to build grid
    #l is half length of squares side
    #s is no of cells along one direction
    s_i, n_i = (n_side/2), n_cell

    #scale to build grid on
    scale = np.linspace(-s_i, s_i, n_i)
    #creating mesh of grid points
    x_x, y_y = np.meshgrid(scale, scale)
    #flattening array of coordinates
    x_f, y_f = x_x.flatten(), y_y.flatten()

    return x_f, y_f

def get_uvn(normal):
    """
    Creates rotation matrix for given normal i.e. calculates direction
    components of axis system with given normal.

    Parameters
    ----------
    normal : ndarray, shape (3,)
        direction components of normal vecctor

    Returns
    -------
    uvn : ndarray, shape (3, 3)
        rotation matrix with u-v vectors for plane and normal
    """

    # Instantiate reference
    prmdir = np.zeros(3)
    refrnc = np.array([1, 0, 0])

    # Initializing output
    uvn = np.zeros((3, 3))

    # Check and update reference vector
    flag = int(np.abs(np.dot(normal, refrnc))>0.7)
    prmdir[flag] = 1

    # Calculating u-v vectors for plane
    u_vec = (prmdir - normal*normal[flag])
    v_vec = np.cross(normal, u_vec)

    #updating output element
    uvn[(1-flag), :] = u_vec
    uvn[flag, :] = v_vec
    uvn[2, :] = normal

    return uvn/LA.norm(uvn, axis=1)[:, None]

def transform_pts(points, origin, normal):

    """
    Trasnforms points from standard axis system to a local axis system

    Parameters
    ----------
    points : ndarray, shape (n, 3)
        points to be transformed
    origin : ndarray, shape (3,)
        cordinates of origin
    normal : ndarray, shape (3,)
        direction components of normal vecctor

    Returns
    -------
    output : ndarray, shape (n, 3)
        transformed point coordinates
    """

    # Calculating local axis
    local_axis = get_uvn(normal)


    return (points - origin) @ local_axis



def get_norm_area(tri):
    """Calculate norma and area of triangle"""
    if tri.ndim==2:
        i0, i1, i2 = 0, 1, 2
    elif tri.ndim==3:
        i0 = (slice(None), 0, slice(None))
        i1 = (slice(None), 1, slice(None))
        i2 = (slice(None), 2, slice(None))
    s_ab = tri[i1] - tri[i0]
    s_bc = tri[i2] - tri[i1]
    norm = np.cross(s_ab, s_bc)
    area = np.linalg.norm(norm, axis=-1)
    return norm, area

def transform_to_image(tris, WINDOW=5):
    """Transforms list of triangles to image"""
    GRID_PTS = np.array([*get_grid_points(WINDOW, 32),
                         np.zeros(1024,)]).T
    INDICES = np.arange(1024)

    xyzm = np.ones(1024)*100
    norm, area = get_norm_area(tris)

    for i in range(len(tris)):
        dist = ((GRID_PTS-tris[i, 0])@norm[i])/(-norm[i, -1])
        xyz_ = np.hstack([GRID_PTS[:, :2], dist[:, None]])

        area_sum = np.zeros_like(dist)
        for idx in [[0, 1], [1, 2], [2, 0]]:
            tri_temp = np.repeat(tris[i, idx, :][None], 1024, axis=0)
            tri_aray = np.concatenate((xyz_[:, None, :], tri_temp), axis=1)
            area_sum += get_norm_area(tri_aray)[1]

        idx = INDICES[np.abs(area_sum-area[i])<1e-5]
        xyzm[idx] = np.min([xyzm[idx], dist[idx]], axis=0)

    xyzm[xyzm>50] = -20
    return xyzm.reshape(32, 32)

def standardize_depth(vtx_pnts: np.ndarray, mtd_dirn: np.ndarray) -> np.ndarray:
    cog_part = np.mean(vtx_pnts, axis=0)
    d_array = np.sum((vtx_pnts - cog_part) * mtd_dirn, axis=1)
    
    upr_band = d_array.max() - np.quantile(d_array, 0.66)
    lwr_band = np.quantile(d_array, 0.33) - d_array.min()
        
    if lwr_band >= upr_band:
        d_array = np.abs(d_array - d_array.max())
    else :
        d_array = np.abs(d_array - d_array.min())
    
    return d_array

def extract_pt_cloud(triangles, point_cloud):
    mask = np.any(np.isin(triangles, point_cloud), axis=1)
    c_tri = triangles[mask]
    c_pnt, _c_tri = np.unique(c_tri, return_inverse=True)
    c_pnt = c_pnt.astype(np.int64)
    c_tri = _c_tri.reshape(c_tri.shape)
    return c_pnt, c_tri

def outlier_treatment(array, m=0.25):
    upr = np.quantile(array, 1-m)
    lwr = np.quantile(array, m)
    bnd = (1 + 2*m) * (upr - lwr)
    ulmt = upr + bnd
    llmt = lwr - bnd
    array = np.where(array>ulmt, ulmt, array)
    array = np.where(array<llmt, llmt, array)
    return array