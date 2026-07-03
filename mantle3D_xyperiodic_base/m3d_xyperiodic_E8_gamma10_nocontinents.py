import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = 4, 4, 1
Nx, Ny, Nz = 128, 128, 64
RaT = 1e4
#RaC = 2.2e4
Le = 100
#h = 0
E = 8
γ = 10
dealias = 3/2
timestepper = d3.RK222
max_timestep = 1e-3
stop_sim_time = max_timestep*1e4
dtype = np.float64
restart=0

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
T = dist.Field(name='T', bases=(xbasis,ybasis,zbasis))
#C = dist.Field(name='C', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis)) # this is 3 unknowns
tau_p1  = dist.Field(name='tau_p1')
tau_p2x = dist.Field(name='tau_p2x')
tau_p2y = dist.Field(name='tau_p2y')
tau_T1 = dist.Field(name='tau_T1', bases=(xbasis,ybasis))
tau_T2 = dist.Field(name='tau_T2', bases=(xbasis,ybasis))
#tau_C1 = dist.Field(name='tau_C1', bases=(xbasis,ybasis))
#tau_C2 = dist.Field(name='tau_C2', bases=(xbasis,ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis)) # this is 3 unknowns
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis)) # this is 3 unknowns

# Substitutions
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_T = d3.grad(T) + ez*lift(tau_T1) # First-order reduction
#grad_C = d3.grad(C) + ez*lift(tau_C1) # First-order reduction
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, T, u, tau_p1, tau_p2x, tau_p2y, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())

# incompressible
problem.add_equation("trace(grad_u) + tau_p1 = 0")
# energy equation
problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = - u@grad(T) + E")
# continent
#problem.add_equation("dt(C) - (1/Le)*div(grad_C) + lift(tau_C2) = - u@grad(C) + γ*(1-C)")
# stokes momentum equation
problem.add_equation("-div(grad_u) + grad(p) - (RaT*T)*ez + lift(tau_u2) + tau_p2x*ex + tau_p2y*ey = 0") # this is 3 equations

# BC: convectively unstable
problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=Lz) = 0")
# BC: continents on top
#problem.add_equation("C(z=0) = 1")
#problem.add_equation("C(z=Lz) = 0")
# BC: rigid lid, rigid bottom
problem.add_equation("ez@u(z=0) = 0")
problem.add_equation("ez@u(z=Lz) = 0")
# BC: stress free top, bottom
problem.add_equation("(ez@grad(u)@ex)(z=0) = 0")
problem.add_equation("(ez@grad(u)@ex)(z=Lz) = 0")
problem.add_equation("(ez@grad(u)@ey)(z=0) = 0")
problem.add_equation("(ez@grad(u)@ey)(z=Lz) = 0")
# BC: gauge conditions, no pressure or mean flow buildup
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(ex@u) = 0")
problem.add_equation("integ(ey@u) = 0")

# 19 unknowns, 19 equations

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep
    T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    T['g'] *= z * (Lz - z) # Damp noise at walls
    T['g'] += 1 - z/Lz # Add linear background
#    C['g'] = 1
else:
    file_handler_mode = 'append'
    write, initial_timestep = solver.load_state('checkpoints_E8_gamma10_nocontinents/checkpoints_s1.h5')

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_E8_gamma10_nocontinents', iter=100, max_writes=10)
snapshots.add_task(T, name='T')
#snapshots.add_task(C, name='C')
snapshots.add_task(ez @ u, name='w')
snapshots.add_task(ex @ d3.curl(u), name='vorticity_x')
snapshots.add_task(ey @ d3.curl(u), name='vorticity_y')
snapshots.add_task(ez @ d3.curl(u), name='vorticity_z')

# Horizontally averaged nonlinear diagnostics
snapshots_nonlinear = solver.evaluator.add_file_handler('snapshots_nonlinear_E8_gamma10_nocontinents', iter=10, max_writes=200)
snapshots_nonlinear.add_task(d3.Average((ez @ u)*T, ('x', 'y')), name='convective_heat_flux_z')
snapshots_nonlinear.add_task(d3.Average((u@u)/2, ('x', 'y', 'z')), name='KE_avg')

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints_E8_gamma10_nocontinents', sim_dt=max_timestep*1000, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.25, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@u)/2, name='KE')

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
