# Example of Script to Run a Simulation

# Create Transfer Matrix
h11 = signal.TransferFunction([-0.19],[1, 0])
h12 = signal.TransferFunction([-1.7],[19.5, 1])
h21 = signal.TransferFunction([-0.763],[31.8, 1])
h22 = signal.TransferFunction([0.235],[1, 0])
H = [[h11, h12], [h21, h22]]
# Horizons
p = 10
m = 3
# Weights
Q = 1*np.eye(p*ny)
R = 10*np.eye(m*nu)
# Constraints
du_max = 0.2
du_min = -0.2
# Sample Time
Ts = 1
controller = GPCController(H, Ts, p, m, Q, R, du_min, du_max)
# Create Simulation Environment
tsim = 100
sim = Simulation(controller, real_model)
sim.run(tsim)
sim.show_results()
