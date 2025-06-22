from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, RigidBody, System, LinearPathway

# ==========================
# Definitions
t = symbols('t')
q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')
ma, mb, g = symbols('m_a m_b g')
Fp1, Fp2 = symbols('Fp1 Fp2')

N = ReferenceFrame('N')
P = ReferenceFrame('P')
Q = ReferenceFrame('Q')

# ==========================
# Points
O = Point('O')
P1 = Point('P1')
P2 = Point('P2')
P3 = Point('P3')

P1.set_pos(O, 0)
P2.set_pos(P1, q2 * N.y + q1 * N.x)
P3.set_pos(P1, q3 * N.x + q4 * N.y)

# ==========================
# Bodies
wall = RigidBody('W', masscenter=P1, frame=N, mass=0)  # Ground
block1 = RigidBody('B1', masscenter=P2, frame=P, mass=ma)
block2 = RigidBody('B2', masscenter=P3, frame=Q, mass=mb)

# ==========================
# System
system = System.from_newtonian(wall)
system.add_bodies(block1, block2)

# Orientations
P.orient_axis(system.frame, 0, system.frame.z)           # No actual angle used
Q.orient_axis(P, 0, P.z)                                  # No actual angle used

# Velocities
block1.masscenter.set_vel(system.frame, u1 * N.x + u2 * N.y)
block2.masscenter.set_vel(system.frame, u3 * N.x + u4 * N.y)

# Gravity
system.apply_uniform_gravity(-g * N.y)

# ==========================
# Generalized coords/speeds
system.add_coordinates(q1, q2, q3, q4)
system.add_speeds(u1, u2, u3, u4)

# Kinematic Differential Equations
system.add_kdes(
    u1 - q1.diff(t),
    u2 - q2.diff(t),
    u3 - q3.diff(t),
    u4 - q4.diff(t),
)

# ==========================
# Springs / Forces
path1 = LinearPathway(P1, P2)
path2 = LinearPathway(P2, P3)

system.add_loads(path1.to_loads(Fp1)[1])
system.add_loads(path2.to_loads(Fp2)[1])

# ==========================
# Finalize
system.validate_system()
eoms = system.form_eoms()


