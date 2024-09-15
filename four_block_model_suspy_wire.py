from sympy.physics.mechanics import (
    dynamicsymbols, ReferenceFrame, Point, RigidBody, Particle, System, Force
)
from sympy import zeros, symbols

import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from IPython.display import display
from suspycious import Model
import suspycious.components as scmp
from wire_suspy import make_wire_suspy

k1, k2, k3, k4, g, c1, c2, t = symbols('k1 k2 k3 k4 g c1 c2 t')

from utils import (
    add_points, get_tension_dirs, give_deltas, give_tensions, 
    get_linpaths, get_wire_dir, apply_force
)
from pathway import LinearPathway

def four_block_model_suspy():
    l1, l2, l3, l4 = symbols('l1 l2 l3 l4', positive=True)

    model2 = Model()
    S = model2.add(scmp.RigidBody("S"))
    B = model2.add(scmp.RigidBody("B"))
    C = model2.add(scmp.RigidBody("C"))
    D = model2.add(scmp.RigidBody("D"))
    F = model2.add(scmp.RigidBody("F"))

    S.set_global_position(0, 0, 0)
    B.set_global_position(0, 0, -l1)
    C.set_global_position(0, 0, -l1 - l2)
    D.set_global_position(0, 0, -l1 - l2 - l3)
    F.set_global_position(0, 0, -l1 - l2 - l3 - l4)

    S.set_global_orientation(0, 0, 0)
    B.set_global_orientation(0, 0, 0)
    C.set_global_orientation(0, 0, 0)
    D.set_global_orientation(0, 0, 0)
    F.set_global_orientation(0, 0, 0)

    # Define geometrical properties (see images)
    d1, d2, n1, n2, w1, w2 = symbols('d1 d2 n1 n2 w1 w2', real=True, positive=True)
    d3, n3, w3 = symbols('d3 n3 w3', real=True, positive=True)
    d4, n4, w4 = symbols('d4 n4 w4', real=True, positive=True)
    d5, n5, w5 = symbols('d5 n5 w5', real=True, positive=True)
    d6, n6, w6 = symbols('d6 n6 w6', real=True, positive=True)
    d7, n7, w7 = symbols('d7 n7 w7', real=True, positive=True)
    d8, d9 = symbols('d8 d9', real=True, positive=True)

    # Adding points to the bodies
    add_points(body=S, point='P', attachment_points=[w1, n1, 0])
    add_points(body=B, point='B', attachment_points=[w1, n1, d1])
    add_points(body=B, point='C', attachment_points=[w2, n2, -d2])
    add_points(body=C, point='D', attachment_points=[w2, n2, d3])
    add_points(body=C, point='E', attachment_points=[w3, n3, -d4])
    add_points(body=D, point='F', attachment_points=[w3, n3, d5])
    add_points(body=D, point='G', attachment_points=[w4, n4, -d6])
    add_points(body=F, point='H', attachment_points=[w4, n4, d7])
    add_points(body=F, point='J', attachment_points=[w5, n5, -d8])

    # Getting tension direction
    dirT11, dirT12, dirT13, dirT14 = get_wire_dir(
        body1=S, points_1='top', body2=B, points_2='top', suspension_body=S
    )
    dirT21, dirT22, dirT23, dirT24 = get_wire_dir(
        body1=B, points_1='bottom', body2=C, points_2='top', suspension_body=S
    )
    dirT31, dirT32, dirT33, dirT34 = get_wire_dir(
        body1=C, points_1='bottom', body2=D, points_2='top', suspension_body=S
    )
    dirT41, dirT42, dirT43, dirT44 = get_wire_dir(
        body1=D, points_1='bottom', body2=F, points_2='top', suspension_body=S
    )

    sb_paths = get_linpaths(body1=S, body2=B, points_1='top', points_2='top', suspension_body=S)
    bc_paths = get_linpaths(B, C, 'bottom', 'top', S)
    cd_paths = get_linpaths(C, D, 'bottom', 'top', S)
    df_paths = get_linpaths(D, F, 'bottom', 'top', S)

    wire_sbs = [
        make_wire_suspy(n=3, linpath=sb_paths, index=i,
                        global_frame=S.global_frame, suspension_body=S) 
        for i in [0, 1, 2, 3]
    ]
    wire_bcs = [
        make_wire_suspy(n=3, linpath=bc_paths, index=i,
                        global_frame=S.global_frame, suspension_body=S)
        for i in [0, 1, 2, 3]
    ]
    wire_cds = [
        make_wire_suspy(n=3, linpath=cd_paths, index=i,
                        global_frame=S.global_frame, suspension_body=S)
        for i in [0, 1, 2, 3]
    ]
    wire_dfs = [
        make_wire_suspy(n=3, linpath=df_paths, index=i,
                        global_frame=S.global_frame, suspension_body=S)
        for i in [0, 1, 2, 3]
    ]

    # Getting deltas
    delta_SB = give_deltas(body1=S, body2=B, suspension_body=S, model=model2)
    delta_BC = give_deltas(body1=B, body2=C, points_1='bottom', points_2='top', suspension_body=S, model=model2)
    delta_CD = give_deltas(body1=C, body2=D, points_1='bottom', points_2='top', suspension_body=S, model=model2)
    delta_DF = give_deltas(body1=D, body2=F, points_1='bottom', points_2='top', suspension_body=S, model=model2)

    masses = [B.M, C.M, D.M, F.M]

    # Getting expression of Tensions
    T11, T12, T13, T14 = give_tensions(n_body=1, k=k1, delta_values=delta_SB, masses=masses)
    T21, T22, T23, T24 = give_tensions(n_body=2, k=k2, delta_values=delta_BC, masses=masses)
    T31, T32, T33, T34 = give_tensions(n_body=3, k=k3, delta_values=delta_CD, masses=masses)
    T41, T42, T43, T44 = give_tensions(n_body=4, k=k4, delta_values=delta_DF, masses=masses)

    force_dict = {
        B: {
            'paths': [sb_paths, bc_paths],
            'tensions': [T11, T12, T13, T14, T21, T22, T23, T24],
            'points': B.points
        },
        C: {
            'paths': [bc_paths, cd_paths],
            'tensions': [T21, T22, T23, T24, T31, T32, T33, T34],
            'points': C.points
        },
        D: {
            'paths': [cd_paths, df_paths],
            'tensions': [T31, T32, T33, T34, T41, T42, T43, T44],
            'points': D.points
        },
        F: {
            'paths': [df_paths],
            'tensions': [T41, T42, T43, T44],
            'points': F.points
        }
    }

    import time
    tic = time.time()
    Sgb = S.global_frame

    apply_force(B, forcedict=force_dict, global_frame=S.global_frame)
    B.Fz = -np.sum([i * g for i in masses])
    apply_force(C, forcedict=force_dict, global_frame=S.global_frame)
    C.Fz = -np.sum([i * g for i in masses[1:]])
    apply_force(D, forcedict=force_dict, global_frame=S.global_frame)
    D.Fz = -np.sum([i * g for i in masses[2:]])
    apply_force(F, forcedict=force_dict, bottom=False, global_frame=S.global_frame)
    F.Fz = -np.sum([i * g for i in masses[3:]])

    tac = time.time()
    print(tac - tic)

    print("Extracting State Space")
    tic = time.time()
    kane = model2.extract_statespace()
    tac = time.time()
    print(tac - tic)
    print("Done")

    A, B = kane.A, kane.B

    L1, L2, L3, L4 = symbols('L1 L2 L3 L4', positive=True)

    A = A.subs({
        -d1 + l1: L1, -d2 - d3 + l2: L2, 
        -d4 - d5 + l3: L3, -d6 - d7 + l4: L4
    })
    B = B.subs({
        -d1 + l1: L1, -d2 - d3 + l2: L2, 
        -d4 - d5 + l3: L3, -d6 - d7 + l4: L4
    })

    #print(A[40, 10])
    
    return A, B, kane

