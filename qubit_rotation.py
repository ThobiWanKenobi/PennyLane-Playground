import pennylane as qp
from jax import numpy as np
import jax
import jaxopt

# Create device which the simulation will run on
# wires parameter determines of subsystems to initialize
# the device with.
dev1 = qp.device("lightning.qubit", wires=1)

# For a python function to be a valid quantum function
# 1. Quantum functions must contain quantum operations, one operation per line, in the order in which they are to be applied.
# 2. Quantum functions must return either a single or a tuple of measured observables.
@qp.qnode(dev1)
def circuit(params):
    # Apply the gate R_x(\phi_1)=e^{-i\phi_1\sigma_x/2}=[\cos\frac{\phi_1}{2} & -i\sin\frac{\phi_1}{2}\\-i\sin\frac{\phi_1}{2} & \cos\frac{\phi_1}{2}]
    qp.RX(params[0], wires=0)

    # Apply the gate R_y(\phi_2)=e^{-i\phi_2\sigma_y/2}=[\cos\frac{\phi_2}{2} & -\sin\frac{\phi_2}{2}\\ \sin\frac{\phi_2}{2} & \cos\frac{\phi_2}{2}]
    qp.RY(params[1], wires=0)

    # Return the expectation value of the Pauli-Z operator
    # Corresponding to:
    # <\psi| \sigma_z |\psi> = <0|R_x(\phi_1)^{\dagger}R_y(\phi_2)^{\dagger}\sigma_zR_y(\phi_2)R_x(\phi_1)|0>=cos(\phi_1)cos(\phi_2).
    return qp.expval(qp.PauliZ(0))

params = np.array([0.54, 0.12])
print(circuit(params))

# Differentiate the circuit function to obtain the 
# gradient using jax. Returns a functions, representing
# the derivative of the QNode with respect to the 
# argument specified in argnums.
dcircuit = jax.grad(circuit, argnums=0)

print(dcircuit(params))

# Optimization
def cost(x):
    return circuit(x)

init_params = np.array([0.011, 0.012])
print(cost(init_params))

# initialise the optimizer
opt = jaxopt.GradientDescent(cost, stepsize=0.4, acceleration = False)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params
opt_state = opt.init_state(params)

for i in range(steps):
    # update the circuit parameters
    params, opt_state = opt.update(params, opt_state)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))