"""Render the IN625 concentration of a graded blade design from its pressure and suction sides.

Uses the vtk module directly and renders off screen, so it needs no display:

    python visualization/render_blade_design.py results/5_blade/data_creep_30/vtk/sol_050.vtu \
        figures/renders/blade_tip_traction_30 [--mesh]

writes <prefix>_pressure.png and <prefix>_suction.png (and <prefix>_mesh.png with --mesh, the
surface mesh in grey). The colour map is ParaView's "Cool to Warm" on [0, 1]; no colour bar is
drawn (visualization/plot_blade_designs.py assembles the figure and adds it).
"""
import argparse
import os

import vtk

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("vtu")
parser.add_argument("prefix")
parser.add_argument("--mesh", action="store_true", help="also write the surface mesh view")
parser.add_argument("--size", type=int, nargs=2, default=(1400, 1400))
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.prefix)), exist_ok=True)

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(args.vtu)
surface = vtk.vtkDataSetSurfaceFilter()
surface.SetInputConnection(reader.GetOutputPort())
surface.Update()

lut = vtk.vtkColorTransferFunction()  # ParaView "Cool to Warm"
lut.SetColorSpaceToDiverging()
lut.AddRGBPoint(0.0, 0.23137254902, 0.298039215686, 0.752941176471)
lut.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
lut.AddRGBPoint(1.0, 0.705882352941, 0.0156862745098, 0.149019607843)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.SetScalarModeToUseCellFieldData()
mapper.SelectColorArray("mat")
mapper.SetLookupTable(lut)
lut.SetVectorModeToComponent()
lut.SetVectorComponent(0)  # first component: IN625
mapper.SetScalarRange(0.0, 1.0)
mapper.SetInterpolateScalarsBeforeMapping(0)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetAmbient(0.25)
actor.GetProperty().SetDiffuse(0.8)

renderer = vtk.vtkRenderer()
renderer.SetBackground(1.0, 1.0, 1.0)
renderer.AddActor(actor)
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetMultiSamples(8)
window.SetSize(*args.size)
window.AddRenderer(renderer)

bounds = surface.GetOutput().GetBounds()
centre = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]


def shoot(name, direction):
    """Look at the blade from `direction` (a vector from the blade to the camera), span upward."""
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*centre)
    camera.SetPosition(*[c + 300.0 * d for c, d in zip(centre, direction)])
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    camera.Zoom(1.1)
    renderer.ResetCameraClippingRange()
    window.Render()
    image = vtk.vtkWindowToImageFilter()
    image.SetInput(window)
    image.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"{args.prefix}_{name}.png")
    writer.SetInputConnection(image.GetOutputPort())
    writer.Write()
    print(f"written {args.prefix}_{name}.png")


# The camera sits slightly above the tip so that the tip face is visible, as in Fig. 8 of the paper.
shoot("pressure", (1.0, 0.3, 0.2))
shoot("suction", (-1.0, 0.3, 0.2))

if args.mesh:
    mapper.ScalarVisibilityOff()
    actor.GetProperty().SetColor(0.85, 0.85, 0.9)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.25, 0.25, 0.45)
    shoot("mesh", (1.0, 0.3, 0.2))
