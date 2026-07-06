import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Ri, Ro = 6, 7
Nphi, Ntheta, Nr = 256, 128, 64
RaT = 1e4
RaC = 1.2e4
Le = 10
h = 0
E = 4
γ = 1
drag = 1e-9
dealias = 3/2
timestepper = d3.SBDF2
max_timestep = 1e-2
stop_sim_time = max_timestep*1e4
dtype = np.float64
mesh = None
restart=0

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
sphere = shell.outer_surface

# Fields
p = dist.Field(name='p', bases=shell)
T = dist.Field(name='T', bases=shell)
C = dist.Field(name='C', bases=shell)
u = dist.VectorField(coords, name='u', bases=shell) # this is 3 unknowns
tau_p  = dist.Field(name='tau_p')
#tau_Lx = dist.Field(name='tau_Lx')
#tau_Ly = dist.Field(name='tau_Ly')
#tau_Lz = dist.Field(name='tau_Lz')
tau_T1 = dist.Field(name='tau_T1', bases=sphere)
tau_T2 = dist.Field(name='tau_T2', bases=sphere)
tau_C1 = dist.Field(name='tau_C1', bases=sphere)
tau_C2 = dist.Field(name='tau_C2', bases=sphere)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere) # this is 3 unknowns
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere) # this is 3 unknowns

# Substitutions
phi, theta, r = dist.local_grids(shell)

er = dist.VectorField(coords, name='er', bases=shell.radial_basis)
er['g'][2] = 1

rvec = dist.VectorField(coords, bases=shell.radial_basis)
rvec['g'][2] = r

#rot_x = dist.VectorField(coords, name='rot_x', bases=shell)
#rot_y = dist.VectorField(coords, name='rot_y', bases=shell)
#rot_z = dist.VectorField(coords, name='rot_z', bases=shell)

# component order is [phi, theta, r]
#rot_x['g'][0] = -r * np.cos(theta) * np.cos(phi)   # phi component
#rot_x['g'][1] = -r * np.sin(phi)                   # theta component
#rot_x['g'][2] = 0                                  # radial component

#rot_y['g'][0] = -r * np.cos(theta) * np.sin(phi)
#rot_y['g'][1] =  r * np.cos(phi)
#rot_y['g'][2] = 0

#rot_z['g'][0] =  r * np.sin(theta)
#rot_z['g'][1] = 0
#rot_z['g'][2] = 0

lift_basis = shell.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_T = d3.grad(T) + rvec*lift(tau_T1) # First-order reduction
grad_C = d3.grad(C) + rvec*lift(tau_C1) # First-order reduction
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction

strain_rate = grad_u + d3.trans(grad_u)
shear_stress_i = d3.angular(d3.radial(strain_rate(r=Ri), index=1))
shear_stress_o = d3.angular(d3.radial(strain_rate(r=Ro), index=1))

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, T, C, u, tau_p, tau_T1, tau_T2, tau_C1, tau_C2, tau_u1, tau_u2], namespace=locals())

# incompressible
problem.add_equation("trace(grad_u) + tau_p = 0")
# energy equation
problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = - u@grad(T) + h*(1-C) + E")
# continent
problem.add_equation("dt(C) - (1/Le)*div(grad_C) + lift(tau_C2) = - u@grad(C) + γ*(1-C)")
# stokes momentum equation
problem.add_equation("-div(grad_u) + grad(p) - (RaT*T - RaC*C)*er + lift(tau_u2) + drag*u = 0") # this is 3 equations

# BC: convectively unstable
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("T(r=Ro) = 0")
# BC: continents on top
problem.add_equation("C(r=Ri) = 1")
problem.add_equation("C(r=Ro) = 0")
# BC: rigid lid, rigid bottom
problem.add_equation("radial(u(r=Ri)) = 0")
problem.add_equation("radial(u(r=Ro)) = 0")
# BC: stress free top, bottom
problem.add_equation("shear_stress_i = 0")
problem.add_equation("shear_stress_o = 0")
# BC: gauge conditions, no pressure or mean flow buildup
problem.add_equation("integ(p) = 0")
#problem.add_equation("integ(rot_x@u) = 0")
#problem.add_equation("integ(rot_y@u) = 0")
#problem.add_equation("integ(rot_z@u) = 0")

# 17 unknowns, 17 equations

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    file_handler_mode = 'overwrite'
    initial_timestep = 1e-4
    T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    T['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
    T['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # Add linear background
    C['g'] = 1
    C['g'][..., -1] = 0
else:
    file_handler_mode = 'append'
    write, initial_timestep = solver.load_state('checkpoints_md9/checkpoints_md9_s1.h5')

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_md9', iter=100, max_writes=10)
snapshots.add_task(T, name='T')
snapshots.add_task(C, name='C')
#snapshots.add_task(ephi @ u, name='u_phi')
#snapshots.add_task(etheta @ u, name='u_theta')
snapshots.add_task(u, name='u')
snapshots.add_task(d3.curl(u), name='vorticity')

# Horizontally averaged nonlinear diagnostics
snapshots_nonlinear = solver.evaluator.add_file_handler('snapshots_nonlinear_md9', iter=10, max_writes=200)

conv_flux = er @ (u*T)
diff_flux = er @ (-grad_T)

snapshots_nonlinear.add_task(conv_flux, name='convective_heat_flux')
snapshots_nonlinear.add_task(diff_flux, name='diffusive_heat_flux')
snapshots_nonlinear.add_task(d3.Integrate((u@u)/2, coords), name='KE_int')

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints_md9', sim_dt=max_timestep*1000, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=1, safety=0.25, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u/2, name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_KE = flow.max('KE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(KE)=%f' %(solver.iteration, solver.sim_time, timestep, max_KE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
