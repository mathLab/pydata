from OCC.BRep import BRep_Tool, BRep_Builder, BRep_Tool_Curve
from OCC.BRepAlgo import brepalgo_IsValid
from OCC.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
from OCC.BRepOffsetAPI import BRepOffsetAPI_FindContigousEdges
from OCC.Display.SimpleGui import init_display
from OCC.GeomConvert import (geomconvert_SurfaceToBSplineSurface,
                             geomconvert_CurveToBSplineCurve)
from OCC.gp import gp_Pnt, gp_XYZ
from OCC.IGESControl import (IGESControl_Controller,
    IGESControl_Reader, IGESControl_Controller_Init,
    IGESControl_Writer)
from OCC.Precision import precision_Confusion
from OCC.ShapeAnalysis import ShapeAnalysis_WireOrder
from OCC.ShapeFix import ShapeFix_ShapeTolerance, ShapeFix_Shell
from OCC.StlAPI import StlAPI_Writer
from OCC.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.TopAbs import (TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FORWARD,
                        TopAbs_SHELL)
from OCC.TopExp import TopExp_Explorer, topexp
from OCC.TopoDS import (topods_Face, TopoDS_Compound, topods_Shell, topods_Edge,
                        topods_Wire, topods, TopoDS_Shape)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class IGESHandler(object):
    @classmethod
    def read(cls, filename):
        controller = IGESControl_Controller()
        controller.Init()

        reader = IGESControl_Reader()
        reader.ReadFile(filename)
        reader.TransferRoots()
        shape = reader.OneShape()

        n_faces = 0
        control_point_position = [0]
        faces_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        mesh_points = np.zeros(shape=(0, 3))

        while faces_explorer.More():
            # performing some conversions to get the right format (BSplineSurface)
            face = topods_Face(faces_explorer.Current())
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face = nurbs_converter.Shape()
            brep_face = BRep_Tool.Surface(topods_Face(nurbs_face))
            bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)

            # openCascade object
            occ_face = bspline_face.GetObject()

            # extract the Control Points of each face
            n_poles_u = occ_face.NbUPoles()
            n_poles_v = occ_face.NbVPoles()
            control_polygon_coordinates = np.zeros(
                shape=(n_poles_u * n_poles_v, 3))

            # cycle over the poles to get their coordinates
            i = 0
            for pole_u_direction in range(n_poles_u):
                for pole_v_direction in range(n_poles_v):
                    print(pole_u_direction, pole_v_direction)

                    control_point_coordinates = occ_face.Pole(
                        pole_u_direction + 1, pole_v_direction + 1)
                    control_polygon_coordinates[i, :] = [
                        control_point_coordinates.X(),
                        control_point_coordinates.Y(),
                        control_point_coordinates.Z()
                    ]
                    i += 1
            # pushing the control points coordinates to the mesh_points array
            # (used for FFD)
            mesh_points = np.append(
                mesh_points, control_polygon_coordinates, axis=0)
            control_point_position.append(control_point_position[-1] +
                                          n_poles_u * n_poles_v)

            n_faces += 1
            faces_explorer.Next()

        return {
            'shape': shape,
            'points': mesh_points,
            'control_point_position': control_point_position
        }

    @classmethod
    def write(cls, filename, data, tolerance=1e-6):

        # cycle on the faces to update the control points position
        # init some quantities
        shape = data.shape
        control_point_position = data.control_point_position
        mesh_points = data.points

        faces_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        n_faces = 0
        compound_builder = BRep_Builder()
        compound = TopoDS_Compound()
        compound_builder.MakeCompound(compound)

        while faces_explorer.More():
            # similar to the parser method
            face = topods_Face(faces_explorer.Current())
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face = nurbs_converter.Shape()
            face_aux = topods_Face(nurbs_face)
            brep_face = BRep_Tool.Surface(topods_Face(nurbs_face))
            bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)
            occ_face = bspline_face.GetObject()

            n_poles_u = occ_face.NbUPoles()
            n_poles_v = occ_face.NbVPoles()

            i = 0
            for pole_u_direction in range(n_poles_u):
                for pole_v_direction in range(n_poles_v):
                    control_point_coordinates = mesh_points[
                        +control_point_position[n_faces], :]
                    point_xyz = gp_XYZ(*control_point_coordinates)

                    gp_point = gp_Pnt(point_xyz)
                    occ_face.SetPole(pole_u_direction + 1, pole_v_direction + 1,
                                     gp_point)
                    i += 1

            # construct the deformed wire for the trimmed surfaces
            wire_maker = BRepBuilderAPI_MakeWire()
            tol = ShapeFix_ShapeTolerance()
            brep = BRepBuilderAPI_MakeFace(occ_face.GetHandle(),
                                           tolerance).Face()
            brep_face = BRep_Tool.Surface(brep)

            # cycle on the edges
            edge_explorer = TopExp_Explorer(nurbs_face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = topods_Edge(edge_explorer.Current())
                # edge in the (u,v) coordinates
                edge_uv_coordinates = BRep_Tool.CurveOnSurface(edge, face_aux)
                # evaluating the new edge: same (u,v) coordinates, but
                # different (x,y,x) ones
                edge_phis_coordinates_aux = BRepBuilderAPI_MakeEdge(
                    edge_uv_coordinates[0], brep_face)
                edge_phis_coordinates = edge_phis_coordinates_aux.Edge()
                tol.SetTolerance(edge_phis_coordinates, tolerance)
                wire_maker.Add(edge_phis_coordinates)
                edge_explorer.Next()

            # grouping the edges in a wire
            wire = wire_maker.Wire()

            # trimming the surfaces
            brep_surf = BRepBuilderAPI_MakeFace(occ_face.GetHandle(),
                                                wire).Shape()
            compound_builder.Add(compound, brep_surf)
            n_faces += 1
            faces_explorer.Next()

        IGESControl_Controller_Init()
        writer = IGESControl_Writer()
        writer.AddShape(compound)
        writer.Write(filename)
