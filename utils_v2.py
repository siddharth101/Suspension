from sympy.physics.mechanics import Point
from sympy import Matrix, symbols
import suspycious.components as scmp
from pathway import LinearPathway
import sympy as sp
import numpy as np


def get_forces(n_body, n_wire_body, spring_constants, masses, wire_dict):

    ks = spring_constants

    wires = {}

    for j in range(4): # 4 is the number of wires per body
         for k in range(n_wire_body): # n_wire_body is the number of sub_wires per body (1 + # of violin mode bodies)
             #print(k+1)
             masses_n = (n_wire_body-1)*4 + 1 # this will be 1 for 0 violin modes, 5 for 1 violin modes and so on
             force = get_force_wire(body=f'body{n_body}', wire=f'wire{j+1}', wire_dict=wire_dict, n_wire=k+1,  k=ks[n_body-1], masses=masses[n_body*masses_n:])[0]
             
             points = get_force_wire(body=f'body{n_body}', wire=f'wire{j+1}',  wire_dict=wire_dict,  n_wire=k+1,  k=ks[n_body-1],
                                    masses=masses[n_body*masses_n:])[1]
             path = get_force_wire(body=f'body{n_body}', wire=f'wire{j+1}',  wire_dict=wire_dict,  n_wire=k+1,  k=ks[n_body-1],
                                  masses=masses[n_body*masses_n:])[2]
             bodies = get_force_wire(body=f'body{n_body}', wire=f'wire{j+1}',  wire_dict=wire_dict,  n_wire=k+1,  k=ks[n_body-1],
                                    masses=masses[n_body*masses_n:])[-1]

             wires[f'wire{k+1}{j+1}'] = {'force':force, 'points':points, 'path':path, 'bodies':bodies}
       
    return wires


def get_tension_dirs(body1, body2, points_1='top', points_2='top', suspension_body=None):
    body1_points = list(body1.points.values())
    body2_points = list(body2.points.values())

    if points_1 == 'top':
        body1_points = body1_points[:4]
    else:
        body1_points = body1_points[4:]

    if points_2 == 'top':
        body2_points = body2_points[:4]
    else:
        body2_points = body2_points[4:]

    S = suspension_body
    ten_dirs = [
        i.pos_from(j).express(S.global_frame).normalize()
        for i, j in zip(body1_points, body2_points)
    ]

    return ten_dirs


def get_wire_dir(body1, body2, points_1='top', points_2='top', suspension_body=None):
    paths = get_linpaths(
        body1, body2, points_1=points_1, points_2=points_2,
        suspension_body=suspension_body
    )['paths']

    S = suspension_body

    dirs = [
        i.attachments[0].pos_from(i.attachments[1]).express(S.global_frame).normalize()
        for i in paths
    ]

    return dirs


def give_deltas(body1, body2, suspension_body, model, points_1='top', points_2='top'):
    body1_points = list(body1.points.values())
    body2_points = list(body2.points.values())

    if points_1 == 'top':
        body1_points = body1_points[:4]
    else:
        body1_points = body1_points[4:]

    if points_2 == 'top':
        body2_points = body2_points[:4]
    else:
        body2_points = body2_points[4:]

    points_ = list(zip(body1_points, body2_points))

    op_point = model.op_point
    sb = suspension_body

    deltas_ = [
        i.pos_from(j).express(sb.global_frame).magnitude() -
        i.pos_from(j).express(sb.global_frame).subs(op_point).magnitude()
        for i, j in points_
    ]
    return deltas_


def add_attachment_points(name, body, position='top', *args, **kwargs):
    points = [Point(f'{name}{j}') for j in range(1, 9)]
    matrix_attachment = Matrix([
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1),
        (1, -1, -1), (1, 1, -1), (-1, -1, -1), (-1, 1, -1)
    ])

    for i in range(4) if position == 'top' else range(4, 8):
        body._unprotect()
        pt = str(points[i])
        dx_pt = body._add_symbol(f"{pt}_x", f"{pt}_x", level=0, real=True)
        dy_pt = body._add_symbol(f"{pt}_y", f"{pt}_y", level=0, real=True)
        dz_pt = body._add_symbol(f"{pt}_z", f"{pt}_z", level=0, real=True)

        attachment_points = [dx_pt, dy_pt, dz_pt]
        matrix_cols = [
            matrix_attachment[i:i+1, 0] * attachment_points[0],
            matrix_attachment[i:i+1, 1] * attachment_points[1],
            matrix_attachment[i:i+1, 2] * attachment_points[2]
        ]
        a = [col[0, 0] for col in matrix_cols]
        body.add_fixed_point(pt, a[0], a[1], a[2])

    body._protect()


def give_tensions(n_body, k, masses, delta_values=None):
    n = n_body - 1
    g = symbols('g')
    masses_ = masses[n:]

    tensions = [sp.Rational(1, 4) * np.sum(masses_) * g + k * i for i in delta_values]

    return tensions


def add_points(body, point, attachment_points):
    points = [Point(f'{point}{j}') for j in range(1, 5)]
    w_, n_, d_ = attachment_points

    body.add_fixed_point(str(points[0]), dx=w_, dy=-n_, dz=d_)
    body.add_fixed_point(str(points[1]), dx=w_, dy=n_, dz=d_)
    body.add_fixed_point(str(points[2]), dx=-w_, dy=-n_, dz=d_)
    body.add_fixed_point(str(points[3]), dx=-w_, dy=n_, dz=d_)


def apply_force(body, forcedict, global_frame, bottom=True):
    top_path = forcedict[body]['paths'][0]['paths']
    top_points = list(forcedict[body]['points'].values())[:4]
    top_tensions = forcedict[body]['tensions'][:4]

    for i in range(4):
        body.body.apply_force(
            top_path[i].to_loads(-top_tensions[i], frame=global_frame)[1][1],
            point=top_points[i]
        )

    if bottom:
        bottom_path = forcedict[body]['paths'][1]['paths']
        bottom_points = list(forcedict[body]['points'].values())[4:]
        bottom_tension = forcedict[body]['tensions'][4:]

        for i in range(4):
            body.body.apply_force(
                bottom_path[i].to_loads(-bottom_tension[i], frame=global_frame)[0][1],
                point=bottom_points[i]
            )


def make_wire_bodies(n, model, top_body, bottom_body, attachment_points, suspension_body):
    global_frame = suspension_body.global_frame
    global_origin = suspension_body.global_origin

    name_ = top_body.body.name[0] + bottom_body.body.name[0]
    num = 4 * n

    names = [f'{name_}{j+1}{i}' for j in range(n) for i in range(1, 5)]
    for name in names:
        model.add(scmp.RigidBody(name))

    top_frame = top_body.frame
    bottom_frame = bottom_body.frame
    bottom_body_bottom_points = list(bottom_body.points)[:4]

    p1, p2 = attachment_points
    d1_ = p1.pos_from(global_origin).dot(global_frame.z).subs(model.op_point)
    d2_ = p2.pos_from(p1).dot(global_frame.z).subs(model.op_point)
    points_wire_coeff = [sp.Rational(1 + i, n + 1) for i in range(n)]

    d_ = [d1_ + i * d2_ for i in points_wire_coeff]

    coeffs = []
    for point in bottom_body_bottom_points:
        coeff_ = [
            bottom_body.points[point].pos_from(bottom_body.body.masscenter)
            .subs(model.op_point).dot(bottom_frame.x),
            bottom_body.points[point].pos_from(bottom_body.body.masscenter)
            .subs(model.op_point).dot(bottom_frame.y)
        ]
        coeffs.append(coeff_)

    coeffs_new = []
    for i in range(len(d_)):
        coeffs_new.append([j + [d_[i]] for j in coeffs])

    coeffs_new = [item for sublist in coeffs_new for item in sublist]
    added_bodies = []
    added_coms = []
    if n>0:
        added_bodies = [i[-1] for i in list(model.components.items())[-num:]]
        added_coms = [i[-1].com for i in list(model.components.items())[-num:]]

    for i in range(num):
        added_bodies[i].set_global_position(coeffs_new[i][0], coeffs_new[i][1], coeffs_new[i][2])
        added_bodies[i].set_global_orientation(0, 0, 0)

    if top_body != suspension_body:
        top_points_ = [top_body.points[i] for i in list(top_body.points)[4:]]
    else:
        top_points_ = [top_body.points[i] for i in list(top_body.points)]

    bottom_points_ = [bottom_body.points[i] for i in list(bottom_body.points)[:4]]
    points_list = top_points_ + added_coms + bottom_points_
    body_list = 4 * [top_body] + added_bodies + 4 * [bottom_body]

    return {'points': points_list, 'bodies': body_list}

def get_points(n, model, num_blocks, bodies, attach_points, suspension_body):
    points_bodies = {}

    # Iterate through the number of blocks and create wire bodies for each
    for i in range(num_blocks):
        points_bodies[f'body{i+1}'] = make_wire_bodies(
            n=n, 
            model=model, 
            top_body=bodies[i][0], 
            bottom_body=bodies[i][1],
            attachment_points=[attach_points[i][0], attach_points[i][1]], 
            suspension_body=suspension_body
        )

    return points_bodies


def get_deltas(points_body, model, suspension_body):
    n = len(points_body['points']) // 4 - 1
    deltas_ = []
    op_point = model.op_point
    sb = suspension_body
    deltas_a = {}
    top_body_name = points_body['bodies'][0].name
    bottom_body_name = points_body['bodies'][-1].name
    delta_body_str = [
        f'del_{top_body_name}_{bottom_body_name}_{i+1}' 
        for i in range(len(points_body['bodies']) - 4)
    ]

    for j in range(4):  # 4 because of 4 wires
        for i in range(n):  # n is wire parts per wire (Depends on violin modes)
            m = 4 * i + j
            attach_points = (points_body['points'][m], points_body['points'][m+4])
            print(attach_points)
            delta = [
                attach_points[0].pos_from(attach_points[1])
                .express(sb.global_frame)
                .magnitude() -
                attach_points[0].pos_from(attach_points[1])
                .express(sb.global_frame)
                .subs(op_point).magnitude()
            ]
            deltas_.append(delta)

    deltas_unwrapped = [item for items in deltas_ for item in items]
    dict_del_body = dict(zip(delta_body_str, deltas_unwrapped))

    return dict_del_body


def deltas_dict(points_body, num_blocks, model, suspension_body):
    delta_ = {}

    for i in range(num_blocks):
        top_body_name = points_body[f'body{i+1}']['bodies'][0].name
        bottom_body_name = points_body[f'body{i+1}']['bodies'][-1].name
        print(f"Calculating deltas for points between bodies {top_body_name} and {bottom_body_name}")
        delta_[f'body{i+1}'] = get_deltas(
            points_body=points_body[f'body{i+1}'], 
            model=model, 
            suspension_body=suspension_body
        )

    return delta_


def get_linpaths(points_list, model):
    points = points_list['points']
    n = len(points) // 4 - 1
    linpaths_ = []

    for j in range(4):
        for i in range(n):
            m = 4 * i + j
            points_ = (points[m], points[m+4])
            print(f"Finding linpath between {points_[0]} and {points_[1]}")
            linpath = LinearPathway(points_[0], points_[1])
            linpaths_.append(linpath)

    bodies = points_list['bodies']
    zipped_bods = []
    for j in range(4):
        for i in range(n):
            m = 4 * i + j
            zipped_bods.append((bodies[m], bodies[m+4]))

    paths_dict = {'paths': linpaths_, 'bodies': zipped_bods}

    return paths_dict


def linpaths_dict(points_body, num_blocks, model):
    linpath_dict = {}

    for i in range(num_blocks):
        linpath = get_linpaths(points_body[f'body{i+1}'], model=model)
        linpath_dict[f'body{i+1}'] = linpath

    return linpath_dict


def add_wire_info(point_body, body, deltas, linpaths):
    num_wires = (len(point_body['points']) - 4) // 4
    deltas_ = list(deltas[body].values())
    linpaths_ = linpaths[body]

    wire_dict = {}

    for j in range(4):
        attach_points_ = []
        attach_bodies_mass = []

        for i in range(num_wires):
            m = 4 * i + j
            attach_points = (point_body['points'][m], point_body['points'][m+4])
            attach_bodies = (
                point_body['bodies'][m].body.mass, 
                point_body['bodies'][m+4].body.mass
            )
            attach_points_.append(attach_points)
            attach_bodies_mass.append(attach_bodies)

        print(attach_points_)

        wire_dict[f'wire{j+1}'] = {
            'points': attach_points_,
            'masses': attach_bodies_mass,
            'deltas': deltas_[num_wires * j:num_wires * j + num_wires],
            'linpaths': linpaths_['paths'][num_wires * j:num_wires * j + num_wires],
            'bodies': linpaths_['bodies'][num_wires * j:num_wires * j + num_wires]
        }

    return wire_dict


def get_wires(num_blocks, point_body, deltas, linpaths):
    wires = {}

    for i in range(num_blocks):
        wire_body = add_wire_info(
            point_body[f'body{i+1}'], 
            body=f'body{i+1}', 
            deltas=deltas, 
            linpaths=linpaths
        )
        wires[f'body{i+1}'] = wire_body

    return wires


def get_force_wire(body, wire, wire_dict, n_wire, k, masses):
    g = symbols('g')
    n = n_wire - 1
    path = wire_dict[body][wire]['linpaths'][n]
    points = [path.attachments[0], path.attachments[1]]

    bodies = [
        wire_dict[body][wire]['bodies'][n][0], 
        wire_dict[body][wire]['bodies'][n][1]
    ]
    #print(masses)
    wire_masses = [i[-1] for i in wire_dict[body][wire]['masses'][:-1]] + [0]
    wire_masses_ = wire_masses[n:]

    delta_wire = wire_dict[body][wire]['deltas'][n_wire - 1]
    extension_force = k * delta_wire

    body_gravity = sp.Rational(1, 4) * np.sum(masses) * g
    wire_gravity = [i * g for i in wire_masses_]

    total_force = [extension_force] + [body_gravity] + wire_gravity
    total_force_ = np.sum(total_force)

    return total_force_, points, path, wire_masses_, bodies



def apply_force_on_body(body_force, suspension_body):
    S = suspension_body

    for i in body_force.keys():
        force = body_force[i]['force']
        path = body_force[i]['path']
        points = body_force[i]['points']
        bodies = body_force[i]['bodies']

        j = 0
        for body in bodies:
            body.body.apply_force(
                path.to_loads(-force, frame=S.global_frame)[j][1], 
                point=points[j]
            )
            j += 1

    return


def apply_gravity(body):
    g = symbols('g')
    body.Fz = -g * body.body.mass

    return

