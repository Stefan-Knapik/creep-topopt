# -*- coding: utf-8 -*-
"""
Abaqus CAE/Python script – Norton Creep (Stress Relaxation)
============================================================
Geometry : 1 x 1 m, plane stress
Mesh     : 2 x 2 CPS4 (4-node bilinear plane-stress quad)
Material : E = 160e3, nu = 0.3
Creep    : Norton power-law  de_eq/dt = A * sigma_eq^n  (A=1e-21, n=3.5)
BCs      : left  ux = uy = 0        (fully fixed)
           right ux = uy = u_right  (mixed displacement BC on right edge)

Steps:
  1. Load  (Static)  – ramp elastic displacement in 1 s
  2. Creep (Visco)   – hold displacement, let creep develop

Run:  abaqus cae noGUI=norton_creep_stress_relaxation_mixed.py

Outputs (current working directory):
  displacements.csv  – nodal U1, U2 vs. time
  stresses.csv       – integration-point stresses vs. time
  strains.csv        – integration-point total & creep strains vs. time
"""

# ── Abaqus modules ────────────────────────────────────────────────────────────
from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import ElemType
import regionToolset
import csv

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
E_mod   = 160e3   # Young's modulus            [Pa]
nu      = 0.3      # Poisson's ratio            [-]
A_cr    = 1.0e-21  # Norton coefficient         [Pa^{-n} s^{-1}]
n_cr    = 3.5      # Norton stress exponent     [-]
m_cr    = 0.0      # time-hardening exponent    [-]  (0 = steady-state Norton)

Lx, Ly  = 1.0, 1.0   # plate dimensions        [m]
nx, ny  = 2, 2        # elements per direction
u_right = 0.002       # prescribed displacement at right edge  [m]

t_load  = 1.0         # elastic ramp duration   [s]
t_creep = 1.0e10      # creep hold duration     [s]

# ── Time integration (Visco / creep step) ────────────────────────────────────
# cetol drives implicit vs explicit integration:
#   cetol=large → forward Euler (explicit), has a stability limit dt_stable ~ 1/(E·A·n·σ^(n-1))
#   cetol=small → backward Euler (implicit), unconditionally stable, larger steps allowed
# Here dt_stable ~ 1e9 s; 32 fixed increments match TOGA's N = 32 run.
dt0_creep = t_creep / 32  # fixed time increment  [s]
dt_min    = dt0_creep
dt_max    = dt0_creep
cetol     = 1.0e30  # max creep strain per increment → explicit integration

# ── Time integration (Load / elastic step) ───────────────────────────────────
dt0_load  = t_load / 10.0   # initial increment for elastic ramp
dt_min_load = 1.0e-8
dt_max_load = t_load

BC_TAG = 'mixedBC'    # tag appended to output file names to mark this BC case

MODEL = 'NortonCreep_' + BC_TAG
JOB   = 'NortonCreep_' + BC_TAG
ODB_PATH = JOB + '.odb'

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Clean slate ───────────────────────────────────────────────────────────────
if MODEL in mdb.models:
    del mdb.models[MODEL]
mdb.Model(name=MODEL, modelType=STANDARD_EXPLICIT)
if 'Model-1' in mdb.models and MODEL != 'Model-1':
    del mdb.models['Model-1']
m = mdb.models[MODEL]

# ── Part ──────────────────────────────────────────────────────────────────────
sk = m.ConstrainedSketch(name='__profile__', sheetSize=2.0)
sk.rectangle(point1=(0.0, 0.0), point2=(Lx, Ly))
part = m.Part(name='Plate', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
part.BaseShell(sketch=sk)
del m.sketches['__profile__']

# ── Material ──────────────────────────────────────────────────────────────────
mat = m.Material(name='NortonMat')
mat.Elastic(table=((E_mod, nu),))
# Power-law (time-hardening) creep:  de_eq/dt = A * sigma_eq^n * t^m
# With m=0 this is the steady-state Norton law.
mat.Creep(table=((A_cr, n_cr, m_cr),))

# ── Section (plane-stress, unit thickness) ────────────────────────────────────
m.HomogeneousSolidSection(name='Sec', material='NortonMat', thickness=1.0)
part.SectionAssignment(
    region=part.Set(faces=part.faces[:], name='AllFaces'),
    sectionName='Sec', offset=0.0,
    offsetType=MIDDLE_SURFACE, offsetField='',
    thicknessAssignment=FROM_SECTION)

# ── Mesh: 2 × 2 CPS4 ─────────────────────────────────────────────────────────
tol = 1.0e-6

# Seed each edge individually so we get exactly nx × ny elements
part.seedEdgeByNumber(
    edges=part.edges.getByBoundingBox(-tol, -tol,    -1, Lx+tol,  tol,    1),
    number=nx, constraint=FIXED)   # bottom
part.seedEdgeByNumber(
    edges=part.edges.getByBoundingBox(-tol, Ly-tol,  -1, Lx+tol,  Ly+tol, 1),
    number=nx, constraint=FIXED)   # top
part.seedEdgeByNumber(
    edges=part.edges.getByBoundingBox(-tol, -tol,    -1,  tol,    Ly+tol, 1),
    number=ny, constraint=FIXED)   # left
part.seedEdgeByNumber(
    edges=part.edges.getByBoundingBox(Lx-tol, -tol,  -1, Lx+tol, Ly+tol, 1),
    number=ny, constraint=FIXED)   # right

# CPS4: 4-node bilinear plane-stress quadrilateral (full integration)
part.setElementType(
    regions=regionToolset.Region(faces=part.faces[:]),
    elemTypes=(
        ElemType(elemCode=CPS4,  elemLibrary=STANDARD,
                 hourglassControl=DEFAULT, distortionControl=DEFAULT),
        ElemType(elemCode=CPS3,  elemLibrary=STANDARD)))   # fallback tri
part.generateMesh()

# ── Assembly ──────────────────────────────────────────────────────────────────
a = m.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
inst = a.Instance(name='Plate-1', part=part, dependent=ON)

# ── Edge sets (on the assembly instance) ─────────────────────────────────────
left_set   = a.Set(name='Left',
    edges=inst.edges.getByBoundingBox(-tol, -tol,    -1,  tol,    Ly+tol, 1))
right_set  = a.Set(name='Right',
    edges=inst.edges.getByBoundingBox(Lx-tol, -tol,  -1, Lx+tol, Ly+tol, 1))
bottom_set = a.Set(name='Bottom',
    edges=inst.edges.getByBoundingBox(-tol, -tol,    -1, Lx+tol,  tol,    1))

# ── Steps ─────────────────────────────────────────────────────────────────────
# Both steps are Visco so CE/CEEQ are valid in a single field output request.
# Creep during the 1 s Load step is negligible at these stress levels.

# Step 1 – Visco: ramp elastic displacement (creep inactive at low stress/time)
m.ViscoStep(name='Load', previous='Initial',
            timePeriod=t_load, nlgeom=OFF,
            initialInc=dt0_load, minInc=dt_min_load, maxInc=dt_max_load,
            cetol=1e30)   # huge cetol → elastic-only behaviour during loading

# Step 2 – Visco: creep hold with fixed displacement
m.ViscoStep(name='Creep', previous='Load',
            timePeriod=t_creep, nlgeom=OFF,
            initialInc=dt0_creep,
            minInc=dt_min,
            maxInc=dt_max,
            cetol=cetol)

# ── Boundary conditions ───────────────────────────────────────────────────────
# Mixed-BC case:
#   Left edge fully fixed   : ux = uy = 0
#   Right edge prescribed   : ux = uy = u_right  (applied in Load step)
#   No bottom BC — both bottom corners are already pinned by the side edges,
#   which eliminates the previous bottom-uy / right-uy conflict at (Lx,0).
m.DisplacementBC(name='BC_Left',   createStepName='Initial',
                 region=left_set,   u1=0.0,     u2=0.0,     ur3=UNSET,
                 amplitude=UNSET, distributionType=UNIFORM, localCsys=None)

# Right-edge displacement applied during the Load step
m.DisplacementBC(name='BC_Right',  createStepName='Load',
                 region=right_set,  u1=u_right, u2=u_right, ur3=UNSET,
                 amplitude=UNSET, distributionType=UNIFORM, localCsys=None)

# ── Field output ──────────────────────────────────────────────────────────────
# Both steps are now Visco, so CE/CEEQ are valid in a single unified request.
del m.fieldOutputRequests['F-Output-1']

m.FieldOutputRequest(name='FO_All', createStepName='Load',
                     variables=('S', 'E', 'U', 'CE', 'CEEQ'),
                     frequency=1)

# Remove default history output (not used here; everything via field output)
if 'H-Output-1' in m.historyOutputRequests:
    del m.historyOutputRequests['H-Output-1']

# ── Create & run job ──────────────────────────────────────────────────────────
if JOB in mdb.jobs:
    del mdb.jobs[JOB]

mdb.Job(name=JOB, model=MODEL, type=ANALYSIS,
        memory=90, memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF,
        userSubroutine='', scratch='', resultsFormat=ODB)  # ODB = abaqusConstants enum

print('Submitting job: ' + JOB)
mdb.jobs[JOB].submit(consistencyChecking=OFF)
mdb.jobs[JOB].waitForCompletion()
print('Job complete. Starting post-processing...')

# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING  →  CSV
# ══════════════════════════════════════════════════════════════════════════════
from odbAccess import openOdb

odb      = openOdb(path=ODB_PATH, readOnly=True)
odb_inst = odb.rootAssembly.instances['PLATE-1']

# Derive component labels from first frame with data (Load step, frame 1+)
def _first_data_frame(step):
    for f in step.frames:
        if 'S' in f.fieldOutputs and len(f.fieldOutputs['S'].values) > 0:
            return f
    return step.frames[-1]

ref = _first_data_frame(odb.steps['Load'])

s_comp  = list(ref.fieldOutputs['S'].componentLabels)
e_comp  = list(ref.fieldOutputs['E'].componentLabels)
ce_comp = (list(ref.fieldOutputs['CE'].componentLabels)
           if 'CE' in ref.fieldOutputs else [])

disp_csv    = 'displacements_' + BC_TAG + '.csv'
stress_csv  = 'stresses_'      + BC_TAG + '.csv'
strain_csv  = 'strains_'       + BC_TAG + '.csv'

with open(disp_csv,   'w', newline='') as fd, \
     open(stress_csv, 'w', newline='') as fs, \
     open(strain_csv, 'w', newline='') as fe:

    dw = csv.writer(fd)
    sw = csv.writer(fs)
    ew = csv.writer(fe)

    # Headers
    dw.writerow(['Step', 'Frame', 'StepTime_s', 'TotalTime_s', 'Node', 'U1_m', 'U2_m'])
    sw.writerow(['Step', 'Frame', 'StepTime_s', 'TotalTime_s', 'Element', 'IP'] + s_comp)
    ew.writerow(['Step', 'Frame', 'StepTime_s', 'TotalTime_s', 'Element', 'IP']
                + e_comp
                + ['CE' + c[1:] for c in ce_comp]
                + ['CEEQ'])

    # Cumulative time offset per step
    step_names  = ['Load', 'Creep']
    time_offset = 0.0

    for step_name in step_names:
        if step_name not in odb.steps:
            continue
        step = odb.steps[step_name]

        for fi, frame in enumerate(step.frames):
            t_step  = frame.frameValue          # time within this step
            t_total = time_offset + t_step
            fo = frame.fieldOutputs

            # ── Displacements (nodal) ─────────────────────────────────────
            if 'U' in fo:
                for v in fo['U'].getSubset(region=odb_inst).values:
                    dw.writerow([step_name, fi, t_step, t_total, v.nodeLabel,
                                 v.data[0], v.data[1]])

            # ── Stresses (integration points) ─────────────────────────────
            if 'S' in fo:
                for v in fo['S'].getSubset(region=odb_inst).values:
                    sw.writerow([step_name, fi, t_step, t_total,
                                 v.elementLabel, v.integrationPoint]
                                + list(v.data))

            # ── Strains (total + creep + CEEQ) ────────────────────────────
            if 'E' not in fo:
                continue

            # Build lookup tables for creep output at this frame
            ce_map   = {}
            ceeq_map = {}
            if 'CE' in fo:
                for v in fo['CE'].getSubset(region=odb_inst).values:
                    ce_map[(v.elementLabel, v.integrationPoint)] = list(v.data)
            if 'CEEQ' in fo:
                for v in fo['CEEQ'].getSubset(region=odb_inst).values:
                    raw = v.data
                    # CEEQ is scalar; stored as float or single-element tuple
                    ceeq_map[(v.elementLabel, v.integrationPoint)] = (
                        float(raw[0]) if hasattr(raw, '__len__') else float(raw))

            for v in fo['E'].getSubset(region=odb_inst).values:
                key  = (v.elementLabel, v.integrationPoint)
                ce   = ce_map.get(key,   [0.0] * max(len(ce_comp), 1))
                ceeq = ceeq_map.get(key, 0.0)
                ew.writerow([step_name, fi, t_step, t_total,
                             v.elementLabel, v.integrationPoint]
                            + list(v.data)   # total strains (all components)
                            + ce             # creep strains
                            + [ceeq])        # equivalent creep strain

        # Advance cumulative clock by this step's total time
        time_offset += step.frames[-1].frameValue

odb.close()
print('Done. Output files:')
print('  ' + disp_csv)
print('  ' + stress_csv)
print('  ' + strain_csv)
