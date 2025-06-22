from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, RigidBody, System, LinearPathway, LinearSpring, LinearDamper, Force, inertia

# Generalized coordinates and speeds
q1, u1, alpha, beta = dynamicsymbols('q1 u1 alpha beta')
t = symbols('t')

# Parameters
k1, c, g, m1, ma, mc, IBzz, l = symbols('k1 c g m1 ma mc IBzz l')

# Frames and Points
N = ReferenceFrame('N')
P = ReferenceFrame('P')
O = Point('O')
wc = Point('wc')
P1 = Point('P1')
P2 = Point('P2')

# Positions
wc.set_pos(O, 0)
P1.set_pos(O, q1 * N.x)
P2.set_pos(P1, -l * P.y)

# Bodies
wall = RigidBody('W', masscenter=wc, frame=N, mass=m1)
block = RigidBody('B', masscenter=P1, frame=N, mass=ma)
compound_pend = RigidBody('C', masscenter=P2, frame=P)

# Inertia
compound_pend.central_inertia = inertia(compound_pend.frame, 0, 0, IBzz)

# System
system = System.from_newtonian(wall)

# Add Bodies
system.add_bodies(block, compound_pend)

# Angular orientation
P.orient_axis(system.frame, alpha, system.frame.z)

# Velocities
O.set_vel(N, 0)
block.masscenter.set_vel(system.frame, u1 * N.x)

# Forces
system.apply_uniform_gravity(-g * N.y)
system.add_loads(Force(block, F * block.x))

# Springs/Dampers
path = LinearPathway(wall.masscenter, block.masscenter)
system.add_actuators(LinearSpring(k1, path), LinearDamper(c, path))

# Generalized Coordinates and Speeds
system.add_coordinates(q1, alpha)
system.add_speeds(u1, beta)

# Kinematic Differential Equations
system.add_kdes(u1 - q1.diff(t), beta - alpha.diff(t))

# Finalization
system.validate_system()
system.form_eoms()

### Matches the solution here
### https://www.12000.org/my_notes/cart_motion/report.htm
### and 
### https://docs.sympy.org/latest/modules/physics/mechanics/examples/multi_degree_freedom_holonomic_system.html
