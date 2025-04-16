from sympy.physics.mechanics import (
    dynamicsymbols, ReferenceFrame, Point, RigidBody, Particle, System, Force
)
from sympy import zeros, symbols
import sympy
import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from IPython.display import display
from suspycious import Model
import suspycious.components as scmp
from pathway import LinearPathway
#from utils import add_points, get_tension_dirs, give_deltas, give_tensions, get_linpaths, get_wire_dir, apply_force
from utils_v2 import (
    add_points, make_wire_bodies, get_points, get_deltas, get_linpaths,
    get_wires, get_forces, deltas_dict, linpaths_dict, add_wire_info,
    get_force_wire, apply_force_on_body, apply_gravity
)

d1, d2, n1, n2, w1, w2 = symbols('d1 d2 n1 n2 w1 w2', real=True, positive=True)
d3, n3, w3 = symbols('d3 n3 w3', real=True, positive=True)
d4, n4, w4 = symbols('d4 n4 w4', real=True, positive=True)
d5, n5, w5 = symbols('d5 n5 w5', real=True, positive=True)
d6, n6, w6 = symbols('d6 n6 w6', real=True, positive=True)
d7, n7, w7 = symbols('d7 n7 w7', real=True, positive=True)
d8, d9 = symbols('d8 d9', real=True, positive=True)

l1, l2, l3, l4, l2a = symbols('l1 l2 l3 l4 l_2a', positive=True)

def set_up_bodies():
    model = Model()
    S = model.add(scmp.RigidBody("S"))
    B = model.add(scmp.RigidBody("B"))
    C = model.add(scmp.RigidBody("C"))
    D = model.add(scmp.RigidBody("D"))
    F = model.add(scmp.RigidBody("F"))

    S.set_global_position(0,0,0)
    B.set_global_position(0,0, -l1-d1)
    C.set_global_position(0,0, -l1 -d1-d2-l2-d3)
    D.set_global_position(0,0, -l1 -d1-d2-l2-d3-d4-l3-d5)
    F.set_global_position(0,0, -l1 -d1-d2-l2-d3-d4-l3-d5-d6-l4-d7)
    S.set_global_orientation(0,0,0)
    B.set_global_orientation(0,0,0)
    C.set_global_orientation(0,0,0)
    D.set_global_orientation(0,0,0)
    F.set_global_orientation(0,0,0)

    add_points(body=S, point='P', attachment_points=[w1, n1, 0])

    add_points(body=B, point='B', attachment_points=[w1, n1, d1])
    add_points(body=B, point='C', attachment_points=[w2, n2, -d2])

    add_points(body=C, point='D', attachment_points=[w2, n2, d3])
    add_points(body=C, point='E', attachment_points=[w3, n3, -d4])

    add_points(body=D, point='F', attachment_points=[w3, n3, d5])
    add_points(body=D, point='G', attachment_points=[w4, n4, -d6])

    add_points(body=F, point='H', attachment_points=[w4, n4, d7])
    add_points(body=F, point='J', attachment_points=[w5, n5, -d8])

    return S, B, C, D, F, model


S, B, C, D, F, model2 = set_up_bodies()
sys = model2.system
n_violin_modes = 1

bodies_ = [[S, B], [B, C], [C, D], [D, F]]
attach_points_ = [[S.P1, B.B1], [B.C1, C.D1], [C.E1, D.F1], [D.G1, F.H1]]


def get_kane( n_violin_modes=n_violin_modes, bodies=bodies_,
    attach_points=attach_points_, model=model2, suspension_body=S):



    points_bodies = get_points(
            n=n_violin_modes, num_blocks=4, model=model,
            bodies=bodies_, attach_points=attach_points_, suspension_body=suspension_body)

    #model2.add_bodies_system()


    deltas = deltas_dict(
        points_body=points_bodies, num_blocks=4, model=model,
        suspension_body=suspension_body
    )

    linpaths = linpaths_dict(
        points_body=points_bodies, num_blocks=4, model=model
    )

    num_blocks = 4
    all_masses = [
        i.body.mass for j in range(num_blocks)
        for i in points_bodies[f'body{j + 1}']['bodies']
    ]

    all_masses_ = list(dict.fromkeys(all_masses))
    wires = get_wires(
        num_blocks=4, point_body=points_bodies,
        deltas=deltas, linpaths=linpaths
    )

    ks = symbols('k1:5')

    n_wire_per_wire = n_violin_modes + 1
    body1_force = get_forces(
            n_body=1, n_wire_body=n_wire_per_wire,
            wire_dict=wires, spring_constants=ks, masses=all_masses_
        )
    body2_force = get_forces(
        n_body=2, n_wire_body=n_wire_per_wire,
        wire_dict=wires, spring_constants=ks, masses=all_masses_
    )
    body3_force = get_forces(
        n_body=3, n_wire_body=n_wire_per_wire,
        wire_dict=wires, spring_constants=ks, masses=all_masses_
    )
    body4_force = get_forces(
        n_body=4, n_wire_body=n_wire_per_wire,
        wire_dict=wires, spring_constants=ks, masses=all_masses_
    )

    for j in [body1_force, body2_force, body3_force, body4_force]:
        for i in j.keys():
            force = j[i]['force']
            path = j[i]['path']
            points = j[i]['points']
            bodies = j[i]['bodies']

            sys.add_loads(path.to_loads(-force, frame=suspension_body.global_frame)[0])
            sys.add_loads(path.to_loads(-force, frame=suspension_body.global_frame)[1])


    apply_gravity(model.components['S'])
    apply_gravity(model.components['B'])
    apply_gravity(model.components['C'])
    apply_gravity(model.components['D'])
    apply_gravity(model.components['F'])

    print("The bodies have been set up, forces applied, now solving for dynamics")

    kane_ = model.extract_statespacesystem()

    return kane_

