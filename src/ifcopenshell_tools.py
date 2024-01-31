from typing import Tuple

import ifcopenshell
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.gp import gp_Mat, gp_Vec, gp_Trsf, gp_Quaternion, gp_Pnt
from ifcopenshell.geom import settings

from pyocc_tools import PyOCCTools

O = 0., 0., 0.
X = 1., 0., 0.
Y = 0., 1., 0.
Z = 0., 0., 1.


# Helper function definitions

# Creates an IfcAxis2Placement3D from Location, Axis and RefDirection specified as Python tuples
def create_ifcaxis2placement(ifcfile, point=O, dir1=Z, dir2=X):
    point = ifcfile.createIfcCartesianPoint(point)
    dir1 = ifcfile.createIfcDirection(dir1)
    dir2 = ifcfile.createIfcDirection(dir2)
    axis2placement = ifcfile.createIfcAxis2Placement3D(point, dir1, dir2)
    return axis2placement


# Creates an IfcLocalPlacement from Location, Axis and RefDirection, specified as Python tuples, and relative placement
def create_ifclocalplacement(ifcfile, point=O, dir1=Z, dir2=X,
                             relative_to=None):
    axis2placement = create_ifcaxis2placement(ifcfile, point, dir1, dir2)
    ifclocalplacement2 = ifcfile.createIfcLocalPlacement(relative_to,
                                                         axis2placement)
    return ifclocalplacement2


# Creates an IfcPolyLine from a list of points, specified as Python tuples
def create_ifcpolyline(ifcfile, point_list):
    ifcpts = []
    for point in point_list:
        point = ifcfile.createIfcCartesianPoint(point)
        ifcpts.append(point)
    polyline = ifcfile.createIfcPolyLine(ifcpts)
    return polyline


def create_sb_shape_from_ifc(
        bound: ifcopenshell.entity_instance) -> TopoDS_Face:
    b_settings = settings()
    b_settings.set(b_settings.USE_PYTHON_OPENCASCADE, True)
    b_settings.set(b_settings.USE_WORLD_COORDS, True)
    b_settings.set(b_settings.EXCLUDE_SOLIDS_AND_SURFACES, False)
    b_settings.set(b_settings.INCLUDE_CURVES, True)
    sore = bound.ConnectionGeometry.SurfaceOnRelatingElement
    if sore.InnerBoundaries is None:
        sore.InnerBoundaries = ()
    try:
        shape = ifcopenshell.geom.create_shape(b_settings, sore)
    except:
        return None
    if bound.RelatingSpace.ObjectPlacement:
        lp = PyOCCTools.local_placement(
            bound.RelatingSpace.ObjectPlacement).tolist()
        mat = gp_Mat(lp[0][0], lp[0][1], lp[0][2], lp[1][0],
                     lp[1][1], lp[1][2], lp[2][0], lp[2][1],
                     lp[2][2])
        vec = gp_Vec(lp[0][3], lp[1][3], lp[2][3])
        trsf = gp_Trsf()
        trsf.SetTransformation(gp_Quaternion(mat), vec)
        shape = BRepBuilderAPI_Transform(shape, trsf).Shape()
    return shape


def calc_point_uv_parameters_on_plane(pnt: gp_Pnt, plane_origin: gp_Pnt,
                                      u_plane: gp_Vec, v_plane: gp_Vec) \
        -> Tuple[float, float]:
    R = gp_Vec(pnt.XYZ()-plane_origin.XYZ())
    u = gp_Vec(u_plane).Dot(R)
    v = gp_Vec(v_plane).Dot(R)
    return u, v
