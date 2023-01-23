#!/usr/bin/env python3

"""
Test module for a training VAE
"""

import numpy as np
import torch
import vtk
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from sklearn.manifold import TSNE
from vtkmodules.vtkCommonColor import vtkNamedColors
# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkFiltersSources import vtkCylinderSource

# noinspection PyUnresolvedReferences
from vtk.util import numpy_support

DEFAULT_VOL_COLOR = (200/255., 200/255., 200/255.)

def make_cylinder_actor():
    colors = vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)

    # This creates a polygonal cylinder model with eight circumferential
    # facets.
    cylinder = vtkCylinderSource()
    cylinder.SetResolution(8)

    # The mapper is responsible for pushing the geometry into the graphics
    # library. It may also do color mapping, if scalars or other
    # attributes are defined.
    cylinderMapper = vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    # The actor is a grouping mechanism: besides the geometry (mapper), it
    # also has a property, transformation matrix, and/or texture map.
    # Here we set its color and rotate it -22.5 degrees.
    cylinderActor = vtkActor()
    cylinderActor.SetMapper(cylinderMapper)
    cylinderActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    cylinderActor.RotateX(30.0)
    cylinderActor.RotateY(-45.0)
    return cylinderActor

def numpy_volume_as_vtk_image_data(source_numpy_array):
    output_vtk_image = vtkImageData()
    output_vtk_image.SetDimensions(source_numpy_array.shape[1], source_numpy_array.shape[0], source_numpy_array.shape[2])

    vtk_type_by_numpy_type = {
        np.uint8: vtk.VTK_UNSIGNED_CHAR,
        np.uint16: vtk.VTK_UNSIGNED_SHORT,
        np.uint32: vtk.VTK_UNSIGNED_INT,
        np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
        np.int8: vtk.VTK_CHAR,
        np.int16: vtk.VTK_SHORT,
        np.int32: vtk.VTK_INT,
        np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
        np.float32: vtk.VTK_FLOAT,
        np.float64: vtk.VTK_DOUBLE
    }
    vtk_datatype = vtk_type_by_numpy_type[source_numpy_array.dtype.type]
    depth_array = numpy_support.numpy_to_vtk(
        source_numpy_array.ravel(), deep=True, array_type=vtk_datatype)
    depth_array.SetNumberOfComponents(1)
    output_vtk_image.GetPointData().SetScalars(depth_array)

    output_vtk_image.Modified()
    return output_vtk_image


def vtk_version_ok(major, minor, build):
    """
    Check the VTK version.

    :param major: Major version.
    :param minor: Minor version.
    :param build: Build version.
    :return: True if the requested VTK version is greater or equal to the actual VTK version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    ver = vtkVersion()
    vtk_version_number = 10000000000 * ver.GetVTKMajorVersion() + 100000000 * ver.GetVTKMinorVersion() \
                             + ver.GetVTKBuildVersion()
    if vtk_version_number >= needed_version:
        return True
    else:
        return False


def get_vtk_volume_to_surface():
    use_flying_edges = vtk_version_ok(8, 2, 0)
    if use_flying_edges:
        try:
            return vtkFlyingEdges3D()
        except AttributeError:
            return vtkMarchingCubes()
    else:
        return vtkMarchingCubes()


def make_volume_actor(volume, iso_value, color=DEFAULT_VOL_COLOR):
    surface = get_vtk_volume_to_surface()
    surface.SetInputData(volume)
    surface.ComputeNormalsOn()
    surface.SetValue(0, iso_value)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor


def make_all_volume_actors(volumes, iso_value, color=DEFAULT_VOL_COLOR):
    actors = []
    for vol in volumes:
        actors.append(make_volume_actor(vol, iso_value, color=color))
    return actors


def initialize_vtk_resourses(windowName=None, background_color=(1., 1., 1.)):
    render_window = vtkRenderWindow()
    if windowName is not None:
        render_window.SetWindowName(windowName)

    renderer = vtkRenderer()
    renderer.SetBackground(background_color)
    render_window.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    return render_window, renderer, interactor

def rgb_hex_to_dec(hex):
    r = int(f"0x{hex[0] + hex[1]}", 16) / 255.
    g = int(f"0x{hex[2] + hex[3]}", 16) / 255.
    b = int(f"0x{hex[4] + hex[5]}", 16) / 255.

    return r, g, b