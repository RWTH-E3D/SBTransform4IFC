"""
    Visualize and Transform IfcOpenShell IfcRelSpaceBoundaries using pythonocc, 
    ifcopenshell and pyqt5.

    Bachelor Thesis:    Modeling approaches in building performance simulation:
                        transformation of the position of IFC space boundaries
    Student: Alex Junck
    Consultant: Veronika Richter, M.Sc. RWTH Aachen
"""

from builtins import print, len
from typing import List
import numpy as np

import ifcopenshell
import ifcopenshell.geom
from OCCUtils.Construct import face_normal, make_offset_shape, scale_uniformal,\
    make_face, make_wire, compound
from OCCUtils.Common import normal_vector_from_plane, minimum_distance, midpoint, \
    intersection_from_three_planes
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeEdge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Iterator, topods_Vertex
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.gp import gp_Mat, gp_Vec, gp_Trsf, gp_Quaternion, gp_Pln, gp_Pnt
from OCC.Display.SimpleGui import init_display

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_PYTHON_OPENCASCADE, True)
settings.set(settings.USE_WORLD_COORDS, True)
settings.set(settings.EXCLUDE_SOLIDS_AND_SURFACES, False)
settings.set(settings.INCLUDE_CURVES, True)


def make_plane(p, ve):
    """
    Inspiration by pythonOCCUtils.Construct.make_plane.
    Creates a plane (gp_Pln) from a given point p (gp_Pnt) and normal vector ve (gp_Vec).
    """
    p = p.add_vec(gp_Vec(0, 0, 0))
    return gp_Pln(p, ve.as_dir())


def translation(shp, vec, copy):
    """
    https://programtalk.com/python-examples/OCC.gp.gp_Trsf/
    Translates any given geometric object shp (TopoDS_Shape) along the vector vec
    (gp_Vec).
    """
    trns = gp_Trsf()  # transformation unit
    trns.SetTranslation(vec)  # transformation set to translation
    brep_trns = BRepBuilderAPI_Transform(shp, trns, copy)
    brep_trns.Build()
    return brep_trns.Shape()


def _a2p(o, z, x):
    """
    Used in computation of local placement
    """
    y = np.cross(z, x)
    r = np.eye(4)
    r[:-1, :-1] = x, y, z
    r[-1, :-1] = o
    return r.T


def _axis2placement(plc):
    """
    Used in computation of local placement.
    """
    z = np.array(plc.Axis.DirectionRatios if plc.Axis else (0, 0, 1))
    x = np.array(plc.RefDirection.DirectionRatios if plc.RefDirection else (1, 0, 0))
    o = plc.Location.Coordinates
    return _a2p(o, z, x)


def local_placement(plc):
    """
    Returns object placement, used for transformation of space boundary positions.
    """
    if plc.PlacementRelTo is None:
        parent = np.eye(4)
    else:
        parent = local_placement(plc.PlacementRelTo)
    return np.dot(_axis2placement(plc.RelativePlacement), parent)


def dump_points(shape, list_points, level=0):
    """
    Inspired by OCCUtils.Topology.DumpTopology
    Returns the points (gp_Pnt) of a TopoDS_Shape in a list. As it takes the vertices
    into account, every point is 2 times in this list: [P0,P1,P1,P2,P2,...,Pn,Pn,P0].
    """
    brt = BRep_Tool()
    s = shape.ShapeType()
    if s == TopAbs_VERTEX:
        pnt = brt.Pnt(topods_Vertex(shape))
        list_points.append(pnt)
    it = TopoDS_Iterator(shape)
    while it.More():
        shp = it.Value()
        it.Next()
        dump_points(shp, list_points, level + 1)
    return list_points


def get_temp_of_space(sp):
    """
    Determines the temperature of a space by a temperature criteria. The temperature
    criteria is set to "SpaceTemperatureMax" by default. It can be changed easily
    (see comment).
    """
    for n in sp.IsDefinedBy:
        if n.RelatingPropertyDefinition.is_a("IfcPropertySet"):
            if n.RelatingPropertyDefinition.Name == "Pset_SpaceThermalRequirements":
                l = n.RelatingPropertyDefinition.HasProperties
                for n1 in l:
                    # enter the required Temperature criteria!
                    if n1.Name == "SpaceTemperatureMax":
                        return n1[2][0]


def get_normal_vector(sh):
    """"
    Determines the normal vector (gp_Vec) of a TopoDS_Shape (every 2nd Level SB is
    plane).
    """
    from OCCUtils.Topology import Topo
    face = list(Topo(sh.bound_shape).faces())[0]  # extracts the face TopoDS_Face
    vecto = face_normal(face)  # gets normal (gp_Dir)
    vecto = gp_Vec(vecto)  # transforms gp_Dir into gp_Vec
    return vecto


def create_edges(list_points, list_edges):
    """
    Creates a list of edges (TopoDS_Edge) from a list of points (gp_Pnt).
    """
    iterator = 0
    while iterator < len(list_points):
        edg = BRepBuilderAPI_MakeEdge(list_points[iterator - 1],
                                      list_points[iterator]).Edge()
        list_edges.append(edg)
        iterator += 1
    return list_edges


def position_from_standard(boundary, standard):
    """
    Returns the factor f for scaling of the translation vector v.
    ||v|| = ||n|| * f * thickness = f * thickness.
    Therefore the factor f implies the position of the SB.
        f = 0.0 : On the building element surface
        f = 0.5 : In the middle of the building element
        f = 1.0 : On the opposite/outside surface of the building element
    Apart from the Standards, this function makes it easy to definer new SB
    transformation.
    """
    if standard == "VDI6020_strict":
        """ Strict differentiation between adiabatic and non adiabatic
            building elements."""
        if boundary.adiabatic or (
                boundary.relatedBuildEle.is_a("IfcSlab")
                and boundary.normal_vec.Z() < 0):
            # generally adiabatic or floor (always)
            return 0.0
        elif (boundary.relatedBuildEle.is_a(
                "IfcSlab") and boundary.relatedBuildEle.PredefinedType == "ROOF") \
                or (boundary.relatedBuildEle.is_a("IfcSlab")
                    and boundary.normal_vec.Z() > 0) \
                or boundary.relatedBuildEle.is_a("IfcRoof"):
            return 1.0
        else:
            return 0.5
    elif standard == "VDI6020_center_all_not_adiabatic":
        """ All building elements are considered non adiabatic."""
        if boundary.relatedBuildEle.is_a("IfcSlab") and boundary.normal_vec.Z() < 0:
            return 0.0
        elif (boundary.relatedBuildEle.is_a(
                "IfcSlab") and boundary.relatedBuildEle.PredefinedType == "ROOF") or (
                boundary.relatedBuildEle.is_a(
                    "IfcSlab") and boundary.normal_vec.Z() > 0)\
                or boundary.relatedBuildEle.is_a("IfcRoof") \
                or (boundary.IntOrExt == "INTERNAL" and boundary.normal_vec.Z() >= 0
                    and boundary.ifc.CorrespondingBoundary.RelatedBuildingElement.is_a(
                    "IfcSlab")):
            return 1.0
        else:
            return 0.5
    elif standard == "VDI2078":
        if boundary.ifc.InternalOrExternalBoundary == "INTERNAL":
            return 0.0
        elif (boundary.relatedBuildEle.is_a(
                "IfcSlab") and boundary.relatedBuildEle.PredefinedType == "BASESLAB"):
            return 0  # external dimensions
        else:  # all external SB are moved to the outside except BaseSlab
            return 1
    elif standard == "DIN12831" or standard == "DINV18599":
        if (boundary.IntOrExt == "EXTERNAL" and boundary.normal_vec.Z() >= 0) \
                or (
                boundary.IntOrExt == "INTERNAL" and boundary.normal_vec.Z() >= 0
                and boundary.ifc.RelatedBuildingElement.is_a(
            "IfcSlab")) \
                or (
                boundary.IntOrExt == "INTERNAL" and boundary.normal_vec.Z() >= 0
                and boundary.ifc.CorrespondingBoundary.RelatedBuildingElement.is_a(
            "IfcSlab")):
            return 1
        elif boundary.IntOrExt == "INTERNAL" and not boundary.normal_vec.Z() < 0:
            return boundary.heated_on_the_other_side
        else:
            return 0
    elif standard == "ASHRAE140":
        if boundary.IntOrExt == "External":
            return 0
        elif boundary.relatedBuildEle.is_a("IfcSlab") and boundary.normal_vec.Z() < 0:
            return 0.0
        elif (
                boundary.IntOrExt == "INTERNAL" and boundary.normal_vec.Z() >= 0
                and boundary.ifc.RelatedBuildingElement.is_a(
            "IfcSlab")) \
                or (
                boundary.IntOrExt == "INTERNAL" and boundary.normal_vec.Z() >= 0
                and boundary.ifc.CorrespondingBoundary.RelatedBuildingElement.is_a(
            "IfcSlab")):
            return 1
        else:
            return 0.5
    else:
        return 0


class SpaceClass:
    """
    Holds all SB-Instances of a space that relevant. Moreover it contains its IFC-Data.
    """
    ifc_type = "IfcSpace"

    def __init__(self, sp):
        self.ifc = sp
        self.list_of_SB = []


class SpaceBoundaryClass:
    """
    Holds IfcRelSpaceBoundary from IfcOpenShell, including a representation of its OCC
    shape and several other relevant attribute for the transformation of their position.
    """
    ifc_type = "IfcRelSpaceBoundary2ndLevel"
    thick = 0.00
    index_all = -1
    index_rel = -1

    def __init__(self, ifc_bound):
        self.adiabatic = 0
        self.heated_on_the_other_side = 0
        self.ifc = ifc_bound
        self.bound_shape = self.get_bound_shape(ifc_bound)
        self.init_points = self.get_ordered_points(self.bound_shape, [])
        self.type = self.ifc.Description
        self.IntOrExt = self.ifc.InternalOrExternalBoundary
        if self.type == "2a":
            self.space = self.ifc.RelatingSpace
        self.newpoints = []
        self.correspSB = self.ifc.CorrespondingBoundary
        if self.type == "2a":
            self.spaceOnTheOtherSide = self.correspSB.RelatingSpace
        self.relatedBuildEle = self.ifc.RelatedBuildingElement
        if self.type == "2a":
            self.thick = self.get_thickness(self.ifc.RelatedBuildingElement)
        self.neighbour_SB = []
        self.normal_vec = gp_Vec()
        self.move_vec = gp_Vec()
        self.init_edges = self.get_edges_of_shape(self.init_points, [])
        self.cutting_planes_orient = []
        self.cutting_planes = []
        self.new_planes_needed = []
        self.move = 0
        self.planes_final_SB = gp_Pln()
        self.vectors = self.get_vectors(self.init_points, [])
        self.new_edges = 0
        self.NEWBOUND = []
        self.additional_cutting_planes = []
        self.aux_points = []
        self.aux_vec_of_edges = []

    @staticmethod
    def get_bound_shape(ifc_bound) -> TopoDS_Shape:
        """
        Computes TopoDS_Shape from an IfcRelSpaceBoundary and transforms it to its
        global position.
        """
        # sore = IfcCurveBoundedPlane
        sore = ifc_bound.ConnectionGeometry.SurfaceOnRelatingElement
        sore.InnerBoundaries = ()  # only ExternalBoundary
        # Settings: OCC = True -> OOC TopoDS_Shape
        shape = ifcopenshell.geom.create_shape(settings,
                                               sore)
        lp = local_placement(ifc_bound.RelatingSpace.ObjectPlacement).tolist()
        mat = gp_Mat(lp[0][0], lp[0][1], lp[0][2], lp[1][0], lp[1][1], lp[1][2],
                     lp[2][0], lp[2][1], lp[2][2])
        vec = gp_Vec(lp[0][3], lp[1][3], lp[2][3])
        trsf = gp_Trsf()
        trsf.SetTransformation(gp_Quaternion(mat),
                               vec)  # moving the shape in its final position
        shape = BRepBuilderAPI_Transform(shape,
                                         trsf).Shape()  # subclass TopoDS_Compound
        return shape

    @staticmethod
    def get_ordered_points(shape, list_points, level=0):
        """
        Inspired by OCCUtils.Topology.DumpTopology
        Returns the points (gp_Pnt) of a TopoDS_Shape in a list. As it takes the
        vertices into a account, every point is 2 times in this list:
        [P0,P1,P1,P2,P2,...,Pn,Pn,P0].
        """
        brt = BRep_Tool()
        s = shape.ShapeType()
        if s == TopAbs_VERTEX:
            pnt = brt.Pnt(topods_Vertex(shape))
            list_points.append(pnt)
        it = TopoDS_Iterator(shape)
        while it.More():
            shp = it.Value()
            it.Next()
            dump_points(shp, list_points, level + 1)
        return list_points

    @staticmethod
    def get_thickness(building_element):
        """
        Inspired by:
        https://stackoverflow.com/questions/50515064/
                        extracting-quantities-of-elements-with-ifcopenshell-in-python
        For building elements with a QuantitySet, the width can be determined with its
        QuantitySet.
        """
        if (not building_element.is_a("IfcBeam")) and (
                not building_element.is_a("IfcDoor")) and (not building_element.
                is_a("IfcWindow")) and (
                not building_element.is_a("IfcBuildingElementProxy")):
            for m in building_element.IsDefinedBy:
                if m.is_a("IfcRelDefinesByProperties"):
                    if m.RelatingPropertyDefinition.is_a("IfcElementQuantity"):
                        if m.RelatingPropertyDefinition[2] == "BaseQuantities":
                            for m1 in m.RelatingPropertyDefinition[5]:
                                if m1.is_a("IfcQuantityLength") and m1[0] == "Width":
                                    return float(m1[3])
        else:
            return 0.00

    @staticmethod
    def get_edges_of_shape(points, edges):
        """
        Creates a list of edges (TopoDS_Edge) from a list of points (gp_Pnt).
        """
        it = 0
        while it < len(points) / 2:
            v = BRepBuilderAPI_MakeEdge(points[it * 2], points[it * 2 + 1]).Edge()
            edges.append(v)
            it += 1
        return edges

    @staticmethod
    def get_vectors(points, list_vectors):
        """
        Creates a list of vectors (gp_Vec) from a list of points (gp_Pnt).
        """
        it = 0
        while it < len(points) / 2:
            v = gp_Vec(points[it * 2], points[it * 2 + 1])
            list_vectors.append(v)
            it += 1
        return list_vectors


class newSBSurfaceClass:
    """
    At first used to store a pair of points with relevant data originating form their
    "Mother-SB".
    Later combined with another pair of points to create a surface representation.
    At the end, instances of this class are transcribed into SpaceBoundaryClass.
    """
    ifc_type = "IfcRelSpaceBoundaryNEW"
    type = "2bNEW"

    def __init__(self, input_plane):
        self.meta = 0
        self.build_elem = 0
        self.other_build_elem = 0
        self.plane = input_plane
        self.normal_vector = gp_Vec()
        self.points = []
        self.space = 0
        self.newBOUND = []
        self.edges = 0


class SBViewer:
    """
    Visualize TopoDS_Shapes from SpaceBoundary Objects.
    Parameters for DisplayShape function:
        [0]: TopoDS_Shape: the actual geometric representation of the SB
        [3]: Color of the geometric element: None, "BLUE", "RED", "WHITE",...
        [4]: Transparency: None = 0 (opaque) to 1.0 (clear)
    Different conditions from the input SB-instance can be used to differentiate between
    SB and colour them accordingly.
    By default, the differentiation is made by their type.
    """

    def __init__(self, bounds: List[SpaceBoundaryClass]):
        display, start_display, add_menu, add_function_to_menu = init_display()
        for bound in bounds:
            if bound.IntOrExt == "EXTERNAL":  # external element transparent
                if bound.type == "2a":
                    display.DisplayShape(bound.NEWBOUND, None, None, None, 0.8)
                elif bound.type == "2b":
                    display.DisplayShape(bound.NEWBOUND, None, None, "BLUE", 0.8)
                else:  # all the added SB-instances
                    display.DisplayShape(bound.NEWBOUND, None, None, "RED", 0.8)
            else:  # internal elements opaque
                if bound.type == "2a":
                    display.DisplayShape(bound.NEWBOUND, None, None, None, 0.5)
                elif bound.type == "2b":
                    display.DisplayShape(bound.NEWBOUND, None, None, "BLUE", 0.5)
                else:  # all the added SB-instances
                    display.DisplayShape(bound.NEWBOUND, None, None, "RED", 0.5)
        print(str(len(bounds)) + " SB shown!")
        display.FitAll()
        start_display()


if __name__ == "__main__":
    """ Enter ifc file here!!"""
    # Set your path to your IFC here. 
    # Tested on FZK-Haus: https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples
    ifc_file = ifcopenshell.open("..\Resources\AC20-FZK-Haus_with_SB.ifc")

    """ Enter applied Standard "VDI6020_strict","VDI6020_center_all_not_adiabatic",
        "VDI2078", "ASHRAE140", "DIN12831", "DINV18599" """
    Norm_Standard = "DIN12831"

    # list with all considered SB-Instances (only Ifc-Data)
    ifc_bounds = ifc_file.by_type("IfcRelSpaceBoundary2ndLevel")
    space_boundaries = []
    # list with all the space-instances
    ifc_spaces = ifc_file.by_type("IfcSpace")
    space_list = []

    final_space_boundaries = []

    """ All IfcRelSpaceBoundary2ndLevel instances are saved as a SpaceBoundaryClass in
        the list space_boundaries. They have a variety of add. attributes compared to
        the IFC data scheme for their transformation"""
    for ifc_bound in ifc_bounds:
        space_boundaries.append(SpaceBoundaryClass(ifc_bound))
    print("All SpaceBoundaryClass-Instances are created!")

    """ All IfcSpace instances are generating an instance of type SpaceClass containing
        additional attributes for performance improvements later on."""
    for ifc_space in ifc_spaces:
        space_list.append(SpaceClass(ifc_space))
    print("All SpaceClass-Instances are created!")

    """ Indexing all SpaceBoundaryClass-Instances for easier and clearer programming."""
    ind = 0
    for sb in space_boundaries:
        sb.index_all = ind
        ind += 1

    """ Separate index for all the relevant SpaceBoundaryClass-Instances (bounding an
        IfcSpace-Instance)."""
    ind = 0
    for sb in space_boundaries:
        if sb.ifc.RelatingSpace.is_a("IfcSpace"):
            sb.index_rel = ind
            ind += 1
    print("Indexing complete!")

    """ Some building elements don't have a QuantitySet: thickness has to be determined
        via geometric relations between the SB -> OCC"""
    print("Remaining thicknesses are being determined...")

    """ Creating a dictionary with the CorrespondingBoundary for faster processing as
        dictionaries are very powerful."""
    id_2_shape = {}
    for sb in space_boundaries:
        if (sb.type == "2a") and (sb.correspSB.RelatingSpace.is_a("IfcSpace")):
            id_2_shape[sb.ifc[0]] = sb.bound_shape

    """ All 2a Sb without ParentingBoundary (2b don't have openings), on the inside (not
        necessarily INTERNAL, but bounding an IfcSpace) are added to an dictionary as
        they might have a thickness."""
    id_2_thickness = {}
    for sb in space_boundaries:
        if (sb.type == "2a") and (sb.ifc.RelatingSpace.is_a("IfcSpace")) and (
                len(sb.ifc.InnerBoundaries)) > 0:
            id_2_thickness[sb.ifc[0]] = sb.thick

    """ Comparing dictionaries for fast determination of RelatedBuildingElement
        thickness. If no value was found via dictionary, the minimum_distance function
        is used in last place, as it is very slow compared to the other methods."""
    for sb in space_boundaries:  # if no thickness determined so far
        if sb.ifc.RelatingSpace.is_a("IfcSpace") and (sb.type == "2a") and (
                sb.thick == 0.00) and (
                not str(sb.ifc.ParentBoundary) == "None"):  # windows, doors,...
            sb.thick = id_2_thickness[sb.ifc.ParentBoundary[0]]
        elif (sb.ifc.RelatingSpace.is_a("IfcSpace")) and (sb.type == "2a") and (
                sb.thick == 0.00):
            corr = id_2_shape[sb.correspSB[0]]
            sb.thick = round(minimum_distance(sb.bound_shape, corr)[0], 2)
    print("...all thicknesses for 2a SB are determined!")

    """ From this point on, all the SB bounding an instance of IfcExternalSpatialElement
        ('external EXTERNAL SB') are no longer relevant for the continuation of this
        program. -> They can be deleted."""
    it = 0
    while it < len(space_boundaries):
        if not space_boundaries[it].ifc.RelatingSpace.is_a("IfcSpace"):
            space_boundaries.pop(it)
        else:
            it += 1
    print("Irrelevant SB deleted!")

    """ Determining if an building element is adiabatic: only for 2a SB"""
    for sb in space_boundaries:
        if sb.adiabatic == 0 and sb.type == "2a":
            if sb.ifc.InternalOrExternalBoundary == "EXTERNAL":
                sb.adiabatic = False
                sb.heated_on_the_other_side = 1
            elif sb.ifc.RelatingSpace == sb.ifc.CorrespondingBoundary.RelatingSpace:
                sb.adiabatic = True
                sb.heated_on_the_other_side = 0.5
            else:
                t1 = get_temp_of_space(sb.space)
                t2 = get_temp_of_space(sb.correspSB.RelatingSpace)
                if t1 == t2:  # same temperature in spaces -> no heat flow
                    sb.adiabatic = True
                    sb.heated_on_the_other_side = 0.5
                else:   # different temperatures in spaces -> heat flow
                    sb.adiabatic = False
                    sb.heated_on_the_other_side = 0.5
                    if t1 > 15 and t2 < 16:  # other room not heated ( <= 15°C)
                        sb.heated_on_the_other_side = 1
                    elif t1 < 16 and t2 > 15:
                        sb.heated_on_the_other_side = 0
    print("Thermal properties have been determined!")

    """ All normal vectors are determined"""
    for sb in space_boundaries:
        sb.normal_vec = get_normal_vector(sb)
    print("Normal vectors have been determined!")

    """ Preliminary definition of magnitude of translation for 2a SB"""
    for sb in space_boundaries:
        if sb.type == "2a":
            sb.move = position_from_standard(sb, Norm_Standard) * sb.thick

    """ Preliminary vectors of translation for 2a SB are calculated"""
    for sb in space_boundaries:
        if sb.type == "2a":
            sb.move_vec = sb.normal_vec.Scaled(sb.move)
    print("Preliminary vectors for 2a SB are ready!")

    """ SB with ParentBoundary (windows, doors,...) are ready to be translated, because
        they always depend on 2a SB and their dimensions don't change. They can be
        copied into another list as they will be only relevant to the end."""
    i = 0
    while i < len(space_boundaries):
        if not str(space_boundaries[
                       i].ifc.ParentBoundary) == "None":  # they are always 2a SB
            space_boundaries[i].bound_shape = translation(
                space_boundaries[i].bound_shape,space_boundaries[i].move_vec, False)
            space_boundaries[i].NEWBOUND = space_boundaries[i].bound_shape
            final_space_boundaries.append(space_boundaries[i])
            space_boundaries.pop(i)
        else:
            i += 1

    for sb in space_boundaries:
        for sp in space_list:
            if sb.ifc.RelatingSpace == sp.ifc:
                sp.list_of_SB.append(sb)

    print("List has been shortened!")

    """ The remaining SB instances in space_boundaries are related to walls, slabs,... .
        The next step is to list for each SB their Neighbour-SB. They have to be listed
        in anti-clockwise order to comply to right-hand rule and facilitate the
        following steps."""
    # counters for verification
    not_found = 0
    explicit = 0
    not_explicit = 0

    for sp in space_list:  # for every space
        for sb in sp.list_of_SB:  # for every SB within that space
            it = 0  # counter-clockwise
            while it < len(sb.init_points) / 2:  # for every edge of that SB
                received = False
                """ --POINT-METHOD--
                    Fastest and but not always suitable."""
                for sb2 in sp.list_of_SB:
                    it2 = 0  # clockwise
                    while it2 < len(sb2.init_points) / 2 and not received:
                        if (gp_Pnt.Distance(sb.init_points[2 * it],
                                            sb2.init_points[2 * it2 + 1]) < 0.005) and (
                                gp_Pnt.Distance(sb.init_points[2 * it + 1],
                                                sb2.init_points[2 * it2]) < 0.005):
                            sb.neighbour_SB.append(sb2)
                            sb.aux_points.append(sb.init_points[2 * it])
                            sb.aux_vec_of_edges.append(sb.vectors[it])
                            received = True
                            explicit += 1
                            break
                        it2 += 1

                if not received:
                    """ --PUZZLE-METHOD--
                        Slower, but more versatile."""
                    dis = sb.vectors[it].Magnitude()
                    reference_pnt = sb.init_points[2 * it]
                    sum_dis = 0
                    temp_dis = sum_dis
                    poi = midpoint(sb.init_points[2 * it], sb.init_points[2 * it + 1])
                    v = scale_uniformal(sb.init_edges[it], poi, 0.994)
                    while sum_dis < dis - 0.0001:
                        temp_dis = sum_dis
                        for sb2 in sp.list_of_SB:
                            if sb.index_all != sb2.index_all:
                                it2 = 0
                                while it2 < len(
                                        sb2.init_points) / 2 and sum_dis < dis - 0.001:
                                    if sum_dis == 0 and \
                                            minimum_distance(sb2.init_edges[it2], v)[
                                                0] < 0.0001 and sb.vectors[
                                        it].CrossMagnitude(sb2.vectors[it2]) < 0.001\
                                            and (sb2.init_points[2 * it2].Distance(
                                                sb2.init_points[2 * it2 + 1]) >=
                                            sb2.init_points[2 * it2].Distance(
                                                sb.init_points[
                                                    2 * it]) or reference_pnt.Distance(
                                        sb2.init_points[2 * it2 + 1]) < 0.005):
                                        reference_pnt = sb2.init_points[2 * it2]
                                        sb.neighbour_SB.append(sb2)
                                        sum_dis += sb2.init_points[2 * it2].Distance(
                                            sb.init_points[2 * it])
                                        sb.aux_points.append(reference_pnt)
                                        sb.aux_vec_of_edges.append(sb.vectors[it])
                                        received = True
                                        if sum_dis < dis - 0.02:
                                            sb.additional_cutting_planes.append(
                                                len(sb.neighbour_SB))
                                            sb.additional_cutting_planes.append(
                                                make_plane(reference_pnt,
                                                           sb.vectors[it]))

                                    if sum_dis > 0 and reference_pnt.Distance(
                                            sb2.init_points[2 * it2 + 1]) < 0.005 and \
                                            sb.vectors[
                                                it].CrossMagnitude(
                                                sb2.vectors[it2]) < 0.001 and \
                                            minimum_distance(sb2.init_edges[it2], v)[
                                                0] < 0.0001:
                                        reference_pnt = sb2.init_points[2 * it2]
                                        sb.neighbour_SB.append(sb2)
                                        sum_dis += sb2.vectors[it2].Magnitude()
                                        sb.aux_points.append(reference_pnt)
                                        sb.aux_vec_of_edges.append(sb.vectors[it])
                                        received = True
                                        if sum_dis < dis - 0.02:
                                            sb.additional_cutting_planes.append(
                                                len(sb.neighbour_SB))
                                            sb.additional_cutting_planes.append(
                                                make_plane(reference_pnt,
                                                           sb.vectors[it]))
                                    it2 += 1
                        if temp_dis == sum_dis:
                            break
                    scale_uniformal(sb.init_edges[it], poi, 1 / 0.994)
                    if (sum_dis + 0.001) > dis and received:
                        explicit += 1
                    elif received:
                        print(
                            "At least one additional Neighbour-SB missing for SB"
                            + str(sb.index_all) + ". Common length along the edge "
                            + str(sum_dis) + "m of " + str(dis) + "m of the edge.")
                        not_explicit += 1
                        if sum_dis == 0 and not received:
                            break
                if not received:
                    print(
                        "No Neighbour-SB detected for an edge of SB with index_all: "
                        + str(sb.index_all))
                    not_found += 1
                it += 1

    total1 = 0
    total2 = 0
    for sb in space_boundaries:
        total1 += len(sb.vectors)
        total2 += len(sb.neighbour_SB)
    print("============================================================================"
        "===========================")
    print("Report - Detection of Neighbour-SB")
    print("Number of edges:                                                  " + str(
        total1))
    print("Number of edges with all Neighbour-SB detected:                   " + str(
        explicit))
    print("Number of edges with at least one Neighbour-SB, but some missing: " + str(
        not_explicit))
    print("Number of edges with no Neighbour-SB detected:                    " + str(
        not_found))
    print("Summary:")
    if total1 == explicit:
        print("All Neighbour-SB (" + str(total2) + ") for every edge have been found!")
    elif not_found == 0:
        print(
            "At least one Neighbour-SB for every edge has been detected! In total "
            + str(total2) +
            " Neighbour-SB have been detected for " + str(total1) + " edges!")
    else:
        print("For " + str(not_found) + " edges out of " + str(total1)
              + ", no Neighbour-SB could be found!")
        print("In total, " + str(total2) + " Neighbour-SB have been detected for "
              + str(total1) + " edges!")
    print("============================================================================"
        "===========================")

    """ Assignment of orientation of the neighbouring SB in the initial state.
        Determining if the neighbouring SB is coplanar (0) or angled (1)"""
    for sb in space_boundaries:
        for sb2 in sb.neighbour_SB:
            if sb.normal_vec.CrossMagnitude(
                    sb2.normal_vec) < 0.01:  # heißt komplanare Ebenen
                sb.cutting_planes_orient.append(0)
            else:
                sb.cutting_planes_orient.append(1)

    """ For 2b SB, no move attribute has been determined as they don't have a distinct
        thickness. Their movement depends on their neighbouring SB."""
    for sb in space_boundaries:
        if sb.type == "2b":
            maxi = 0
            it = 0
            while it < len(sb.neighbour_SB):
                if sb.cutting_planes_orient[it] == 0 and sb.neighbour_SB[it].move >\
                        maxi:
                    maxi = sb.neighbour_SB[it].move
                it += 1
            sb.move = maxi

    """ 2nd time as 2b SB might influence each other and the order in the list
        space_boundaries is not distinct."""
    for sb in space_boundaries:
        if sb.type == "2b":
            maxi = 0
            it = 0
            while it < len(sb.neighbour_SB):
                if sb.cutting_planes_orient[it] == 0\
                        and sb.neighbour_SB[it].move > maxi:
                    maxi = sb.neighbour_SB[it].move
                it += 1
            sb.move = maxi

    """For some 2a SB, their thickness is disturbed by overlapping SB: Beam in wall,
        slab in wall,... """
    for sb in space_boundaries:
        if sb.type == "2a" and sb.normal_vec.Z() == 0\
                and len(sb.ifc.InnerBoundaries) > 0\
                and (sb.ifc.RelatedBuildingElement.is_a("IfcBeam")
                or (sb.ifc.RelatedBuildingElement.is_a("IfcSlab"))):
            maxi = sb.move
            it = 0
            while it < len(sb.neighbour_SB):
                if sb.cutting_planes_orient[it] == 0 and sb.neighbour_SB[it].move >\
                        maxi:
                    maxi = sb.neighbour_SB[it].move
                it += 1
            sb.move = maxi

    """ Checking if a SB and its neighbouring SB are angled (0), coplanar and translated
        the same amount (1) or coplanar an translated differently (2)."""
    for sb in space_boundaries:
        it = 0
        while it < len(sb.cutting_planes_orient):
            if sb.cutting_planes_orient[it] == 1:
                # angle between planes
                sb.new_planes_needed.append(0)
            elif sb.cutting_planes_orient[it] == 0 \
                    and sb.move == sb.neighbour_SB[it].move:
                # coplanar planes, but in the end no new surface created
                sb.new_planes_needed.append(1)
            else:
                # coplanar planes and different translation -> new surface/SB required
                sb.new_planes_needed.append(2)
            it += 1

    """ All normal vectors are determined for the final translation"""
    for sb in space_boundaries:
        sb.move_vec = sb.normal_vec.Scaled(sb.move)

    """ Creation of the planes for the final SB (the final SB will lie in these planes).
        (apart from additional SB-instances)"""
    for sb in space_boundaries:
        pt = sb.init_points[0].Translated(sb.move_vec)
        sb.planes_final_SB = make_plane(pt, sb.normal_vec)

    """ Lists the elements for the additional SB-Instances."""
    new_SB_planes = []
    new_SB_buildelement = []
    new_SB_other_buildelement = []
    new_SB_space = []
    new_SB_meta = []

    """ The cutting planes are determined. They will be used to generate the edges of
        the new SB resp. intersect with 2 other planes to generate the corners. In case
        a new SB is required, basic information is listed to be conveyed to instances
        of the new class newSBSurface later on."""
    for sb in space_boundaries:
        it = 0
        while it < len(sb.new_planes_needed):
            if sb.new_planes_needed[it] == 0:  # angled SB
                sb.cutting_planes.append(sb.neighbour_SB[it].planes_final_SB)
                # corresponds to the translated neighbouring plane
            else:  # coplanar SB
                vector = sb.aux_vec_of_edges[it].Crossed(sb.normal_vec)
                sb.cutting_planes.append(make_plane(sb.aux_points[it], vector))
                if sb.new_planes_needed[it] == 2:  # new SB required
                    new_SB_planes.append(make_plane(sb.aux_points[it], vector))
                    new_SB_buildelement.append(sb.ifc.RelatedBuildingElement)
                    new_SB_other_buildelement.append(sb.neighbour_SB[it].ifc.
                                                     RelatedBuildingElement)
                    new_SB_space.append(sb.ifc.RelatingSpace)
                    new_SB_meta.append(sb.ifc)
                    """ META contains information of its originating neighbouring SB
                        -> not all information is correct.
                        for ex.: the global ID should be different, as it will be
                                 another SB instance,..."""
            it += 1

    for sb in space_boundaries:
        count = 0
        while len(sb.additional_cutting_planes) > 0:
            sb.cutting_planes.insert(sb.additional_cutting_planes[0] + count,
                                     sb.additional_cutting_planes[1])
            sb.new_planes_needed.insert(sb.additional_cutting_planes[0] + count, 0)
            sb.cutting_planes_orient.insert(sb.additional_cutting_planes[0] + count, 1)
            sb.additional_cutting_planes.pop(0)
            sb.additional_cutting_planes.pop(0)
            count += 1

    """ Instances of the new class newSBSurface are created with the listed data
        from above."""
    new_Inst = []
    for nE in new_SB_planes:
        new_Inst.append(newSBSurfaceClass(nE))

    it = 0
    while it < len(new_Inst):
        new_Inst[it].build_elem = new_SB_buildelement[it]
        new_Inst[it].other_build_elem = new_SB_other_buildelement[it]
        new_Inst[it].space = new_SB_space[it]
        new_Inst[it].meta = new_SB_meta[it]
        new_Inst[it].plane = new_SB_planes[it]
        new_Inst[it].normal_vector = normal_vector_from_plane(new_Inst[it].plane)
        it += 1

    """ Consecutive coplanar planes must be deleted from the list, taking into account
        the planes needed to create the points for the add. SB-Instances."""
    newSBcount = 0
    for sb in space_boundaries:
        it = 0
        while it < len(sb.cutting_planes):
            if normal_vector_from_plane(
                    sb.cutting_planes[it - 1]).Crossed(
                normal_vector_from_plane(sb.cutting_planes[it])).Magnitude() < 0.0001:
                sb.cutting_planes.pop(it - 1)
                if sb.new_planes_needed[it - 1] == 2:
                    new_Inst.pop(newSBcount)
                sb.new_planes_needed.pop(it - 1)
                print("Coplanar plane for SB " + str(sb.index_all) + " deleted!")
            else:
                it += 1
                if sb.new_planes_needed[it - 1] == 2:
                    newSBcount += 1

    """ Generating the points for the final SB and add. SB-Instaces (in first place,
        they are in pairs)."""
    newSBcount = 0
    for sb in space_boundaries:
        it = 0
        while it < len(sb.cutting_planes):
            if sb.new_planes_needed[it] == 0 or sb.new_planes_needed[it] == 1:
                inter = intersection_from_three_planes(sb.planes_final_SB,
                                                       sb.cutting_planes[it - 1],
                                                       sb.cutting_planes[it])
                sb.newpoints.append(inter)
                if sb.new_planes_needed[it - 1] == 2:
                    new_Inst[newSBcount].points.append(inter)
                    newSBcount += 1
            elif sb.new_planes_needed[it] == 2:
                print("For new SB-Instance " + str(newSBcount) + " from SB " +
                      str(sb.index_all))
                inter = intersection_from_three_planes(sb.planes_final_SB,
                                                       sb.cutting_planes[it - 1],
                                                       sb.cutting_planes[it])
                sb.newpoints.append(inter)
                print(inter)
                new_Inst[newSBcount].points.append(inter)
            it += 1

    """ Deleting consecutive identical points (for the existing SB-Instances)."""
    for sb in space_boundaries:
        it = 0
        while it < len(sb.newpoints):
            if gp_Pnt.Distance(sb.newpoints[it - 1], sb.newpoints[it]) < 0.00001:
                sb.newpoints.pop(it)
            else:
                it += 1

    """ Creating the new surface areas of the existing SB-Instances."""
    for sb in space_boundaries:
        if len(sb.newpoints) > 1:
            # print(sb.index_all)
            temp = create_edges(sb.newpoints, [])
            sb.new_edges = temp
            temp2 = make_wire(temp)
            temp3 = make_face(temp2)
            sb.NEWBOUND = temp3

    """ Testing if the created point-pairs for the additional SB-Instances are correct.
        These are necessary conditions, but not sufficient conditions!"""
    check = True
    for nE in new_Inst:
        if len(nE.points) != 2:
            check = False
            break
    if len(new_Inst) == 0:
        print("No check for new SB-Instances required!")
    elif len(new_Inst) % 2 == 0 and check:
        print("Good! Even number of point-pairs for new SB-Instances!")
    elif check:
        print("Problem! Odd number of point-pairs for new SB-Instances!")
    else:
        print("Problem! Odd number of instances and not all the instances "
              "contain 2 points.")

    """Determining maximum translation magnitude."""
    max_move = 0
    for sb in space_boundaries:
        for sb2 in sb.neighbour_SB:
            if abs(sb.move - sb2.move) > max_move \
                    and sb.normal_vec.CrossMagnitude(sb2.normal_vec) < 0.00001:
                max_move = abs(sb.move - sb2.move) + 0.00001

    """ Comparing and combining point-pairs for later creation of the surface areas of
        the add. SB-Instance."""
    it = 0
    while it < len(new_Inst):
        it2 = it + 1
        while it2 < len(new_Inst):
            min_dis = 10000
            check = False
            for p in new_Inst[it].points:
                for p2 in new_Inst[it2].points:
                    min_dis = min(min_dis, p.Distance(p2))
            if min_dis == 0:
                min_dis = 1000
            it3 = 0
            while not check and it3 < len(space_boundaries):
                if new_Inst[it].space == space_boundaries[it3].ifc.RelatingSpace:
                    it4 = -1
                    while not check and it4 < len(space_boundaries[it3].newpoints) - 1:
                        for p in new_Inst[it].points:
                            if p.Distance(space_boundaries[it3].newpoints[it4])\
                                    <= 0.0001:
                                for p2 in new_Inst[it2].points:
                                    if p2.Distance(space_boundaries[it3].
                                                    newpoints[it4 - 1]) <= 0.0001 \
                                            or p2.Distance(space_boundaries[it3].
                                                                   newpoints[
                                                               it4 + 1]) <= 0.0001:
                                        check = True
                                        break
                        it4 += 1
                it3 += 1
            if check and new_Inst[it].space == new_Inst[it2].space \
                    and len(new_Inst[it].points) > 0 and len(new_Inst[it2].points) \
                    and new_Inst[it].build_elem == new_Inst[it2].other_build_elem \
                    and new_Inst[it].plane.Distance(new_Inst[it2].plane) <= 0.01 \
                    and new_Inst[it].normal_vector.CrossMagnitude(
                new_Inst[it2].normal_vector) < 0.00000001 \
                    and min_dis <= max_move + 0.0001:
                # fusing 2 point-pairs
                new_Inst[it].points += new_Inst[it2].points
                new_Inst.pop(it2)
                print("Fusion!")
                break
            it2 += 1
        it += 1
    present = len(space_boundaries)

    """ Deleting consecutive identical points (for the add. SB-Instances)."""
    for sb in new_Inst:
        it = 0
        while it < len(sb.points):
            if gp_Pnt.Distance(sb.points[it - 1], sb.points[it]) < 0.00001:
                sb.points.pop(it)
            else:
                it += 1

    """ Creating the surface areas of the add. SB-Instances.
        First, the points for an add. SB-Instance are checked, it they lie in one plane.
        If not, the make_face function will fail. (Precaution)"""
    for sb in new_Inst:
        if len(sb.points) > 2:
            it = 2
            valid = True
            v = gp_Vec(sb.points[0], sb.points[1]).Crossed(gp_Vec(sb.points[1],
                                                                  sb.points[2]))
            while valid and it < len(sb.points) - 1:
                if not v.CrossMagnitude(
                        gp_Vec(sb.points[it - 1], sb.points[it]).Crossed(
                            gp_Vec(sb.points[it], sb.points[it + 1]))) < 0.00001:
                    valid = False
                else:
                    it += 1
            if valid:
                temp = create_edges(sb.points, [])
                sb.edges = temp
                temp2 = make_wire(temp)
                try:
                    temp3 = make_face(temp2)
                except AssertionError:
                    print(f"Assertion Error, Failed to produce face from "
                          f"points {sb.points}. This surface is skippend. "
                          f"This may result in a gap in the resulting model.")
                    continue
                sb.newBOUND = temp3
                sb.normal_vector = normal_vector_from_plane(sb.plane)
            else:
                sb.newBOUND = []
        else:
            sb.newBOUND = []

    """ Adding the add. SB-Instances to the main list."""
    for sb in new_Inst:
        space_boundaries.append(SpaceBoundaryClass(sb.meta))

    """ Setting the type-attribute to "sbNEW" -> the add. SB-Instances can be detected,
        as not all their attributes are correct."""
    it = present
    while it < len(space_boundaries):
        space_boundaries[it].NEWBOUND = new_Inst[it - present].newBOUND
        space_boundaries[it].type = "2bNEW"
        it += 1

    """ Combining the main list of SB (walls, slabs,..., add Instances) with the list
        of SB from windows, doors,... . All the surfaces an instance of a subclass of
        TopoDS_Shape."""
    final_list_with_SB = final_space_boundaries + space_boundaries

    """ TopoDS_Face converted into TopoDS_Compound! -> all the SB surface areas are
        a TopoDS_Compound."""
    for sb in final_list_with_SB:
        if sb.NEWBOUND != []:
            if sb.NEWBOUND.ShapeType() == 4:
                temp = []
                temp.append(sb.NEWBOUND)
                sb.NEWBOUND = compound(temp)
        else:
            sb.NEWBOUND = compound([])

    print("Generating graphical representation of " + str(len(final_list_with_SB))
          + " SBs.")
    SBViewer(final_list_with_SB)
