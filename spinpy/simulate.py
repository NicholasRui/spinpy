from spinpy import objects as ob
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy.physics.quantum import TensorProduct
import qutip as qu
import time
import pdb

# simulate_dynamics_me might be deprecated
def simulate_dynamics_me(H, initial_state, duration, steps=200, mode='states'):
    """ A function to simulate the dynamics of a spin system given initial state and Hamiltonian using Master equation.

    == IN ==
    ** initial_state, qu.qobj: Ket vector representing initial state as a qutip qobj
    ** H, qu.qobj: Hamiltonian operator as a qutip qobj
    ** duration, float: Integration time interval in us
    ** steps, int: Number of timestep
    ** mode, str: What information should be returned by the simulation?
        > 'states' - solver object returns states
        > 'samestate' - solver object returns expectation values for all spins having the same state

    == OUT ==
    ** solver, qu.solver: QuTiP solver object which encodes information about run

    """
    # First, check to make sure that the ket and H are on the same size Hilbert space
    if (H.N != initial_state.N) or (H.s != initial_state.s):
        raise ValueError('H and initial_state are not the same shape. Operators act on different Hilbert spaces.')

    # Check if the mode entered is valid
    mode_list = ['states', 'samestate']
    if not mode in mode_list:
        raise ValueError('Invalid mode specified.')

    e_ops = []
    if mode == 'states':
        pass
    elif mode == 'samestate':
        for ii in range(int(2*H.s+1)):
            basis_state = qu.basis(int(2*H.s+1), ii)
            for jj in range(H.N):
                if jj == 0:
                    collapse_op = basis_state * basis_state.dag()
                else:
                    collapse_op = qu.tensor(collapse_op, basis_state * basis_state.dag())
            e_ops.append(collapse_op)

    # Run simulation
    # Pass option to allow 2000 time steps
    options = qu.Options(nsteps=1e9) #some large number
    solver = qu.mesolve(H.operator, initial_state.rho, tlist=np.linspace(0,duration,steps), e_ops=e_ops, options=options)

    return solver

def GRAPE_pulse(H, duration, N, gamma=0.01, thres=1e-8, init_state=None, autostop=False):
    """ Main function to optimize pulse with GRAPE.
    ** H, qu.qobj: Hamiltonian operator as a qutip qobj
    ** duration, float: Time length of the pulse
    ** N, int: Number of time slices to split Ex and Ey into
    ** gamma, float: Learning parameter, at each step, simulator jumps by gamma * del f
    ** thres, float: Tolerance threshold for final converged run.
    ** autostop, bool: If True, terminate the run after 10000 iterations

    == OUT ==

    """
    ################################################
    # Setup
    ################################################
    print('Setting up run.')

    # Construct basis vectors and collapse operators for Hilbert space
    b_vec = []
    e_ops = []

    # In the future, allow for the implementation of spin ensembles with different
    # spins
    for ii in range(int(2*H.s+1)):
        basis_state = ob.basis(int(2*H.s+1),ii)#qu.basis(int(2*H.s+1), ii)
        for jj in range(H.N):
            if jj == 0:
                basis_vec = basis_state
                collapse_op = TensorProduct(basis_state, basis_state)#np.outer(basis_state, basis_state)#basis_state * basis_state.dag()
            else:
                basis_vec = TensorProduct(basis_vec, basis_state)#np.kron(basis_vec, basis_state)#qu.tensor(basis_vec, basis_state)
                collapse_op = TensorProduct(collapse_op, basis_state.T * basis_state)#np.kron(collapse_op, np.outer(basis_state, basis_state))#qu.tensor(collapse_op, basis_state * basis_state.dag())
        b_vec.append(basis_vec)
        e_ops.append(collapse_op)

    X_control = np.array(sym.symbols('x0:{0}'.format(N), real=True))
    Y_control = np.array(sym.symbols('y0:{0}'.format(N), real=True))

    U = H.unitary(X_control, Y_control, duration)

    # Create U lambda for convenience
    U_lambda = sym.lambdify(np.append(X_control, Y_control), np.array(U))

    ################################################
    # Build desired unitary
    ################################################
    print('Building desired unitary.')
    # Construct the desired unitary. For now, pick the one that swaps all of them down
    for ii in range(H.N):
        if ii == 0:
            Uideal = 2 * ob.spin_operator(0.5,'x')#qu.sigmax()
        else:
            Uideal = TensorProduct(Uideal, 2 * ob.spin_operator(0.5,'x'))#np.kron(Uideal, 2 * ob.spin_operator(0.5,'x'))#qu.tensor(Uideal, qu.sigmax())
    # Construct the ideal bra vectors
    bras_ideal = [sym.conjugate((Uideal * b_vec[ii]).T) for ii in range(len(b_vec))] #[b_vec[ii].dag() * Uideal.dag() for ii in range(len(b_vec))]

    ################################################
    # Calculate fidelity function
    ################################################
    print('Calculating fidelity.')
    if init_state == None:
        F = 0
        for ii in range(len(b_vec)):
            # Calculate <n|Uideal * Ugrape|n>
            term = (bras_ideal[ii] * U * b_vec[ii])[0,0]
            print(ii+1)
            F += term

        # Get fidelity, convert to fast lambda function
        F = sym.conjugate(F) * F / len(b_vec)**2
    else:
        if init_state.pure:
            desired_state = Uideal * init_state.vector
            actual_state = U * init_state.vector

            F = (sym.conjugate(desired_state.T) * actual_state)[0,0]
            F = (sym.conjugate(F) * F)
        else:
            raise ValueError('Error: Mixed state currently does not work. Aborting.')
            desired_state = Uideal * init_state.rho * sym.conjugate(Uideal.T)
            actual_state = U * init_state.rho * sym.conjugate(U.T)

            P, D = desired_state.diagonalize()
            sqrt_desired =  P * D ** (1/2) * P ** -1
            full = sqrt_desired * actual_state * sqrt_desired
            pdb.set_trace()
            P, D = full.diagonalize(); print('6')
            F = sym.trace( D ** (1/2) ) ** 2; print('7')

    F_func = sym.lambdify(np.append(X_control, Y_control), F)

    # Calculate the analytical gradient in both quadratures
    delF_x = []
    delF_y = []

    print('Calculate analytical gradient')
    for ii in range(N):
        diff_x = sym.lambdify(np.append(X_control, Y_control),
                               sym.diff(F, X_control[ii]))
        diff_y = sym.lambdify(np.append(X_control, Y_control),
                               sym.diff(F, Y_control[ii]))
        delF_x.append(diff_x)
        delF_y.append(diff_y)
        print(ii+1)
    print('Done.')

    ################################################
    # Initialize guess and simulation arrays
    ################################################
    print('Initialize guess.')
    # Dumb initial guess, uniform driving with no normalization whatsoever
    current_EpsX = 2 * np.pi * np.ones(N) / duration
    current_EpsY = np.zeros(N)

    # Look at initial Uideal to see if it's reasonable
    # Unfortunately, to evaluate, we need to iterate through all the values we want and cast
    # them as default Python floats (instead of numpy.float64)
    args = []
    for ii in range(len(current_EpsX)):
        args.append(float(current_EpsX[ii]))
    for ii in range(len(current_EpsY)):
        args.append(float(current_EpsY[ii]))

    #Uinit = sym.Matrix(U_lambda(*current_EpsX.astype(float), *current_EpsY.astype(float)))
    Uinit = sym.Matrix(U_lambda(*args))

    N_iter = 1
    F_list = [0] # Initialize list of fidelities
    EpsX_list = []
    EpsY_list = []

    ################################################
    # The fun bit
    ################################################
    print('Starting run.')
    init_trial = True # If this is the first run, delete the first element of F_list

    while ((1 - F_list[-1]) > thres):
        if init_trial: # Hokey bodge to make the while loop run the first time
            F_list = []
            init_trial = False

        start = time.time()

        # Numerically evaluate the fidelity and gradient at the trial point
        control_args = list(np.append(current_EpsX, current_EpsY))
        for ii in range(len(control_args)):
            control_args[ii] = complex(control_args[ii])

        # Calculate trial fidelity. Take real fidelity
        F_trial = np.real(F_func(*control_args))#F.xreplace(param_dict)

        # Append everything to arrays
        F_list.append(float(F_trial))
        EpsX_list.append(np.copy(current_EpsX))
        EpsY_list.append(np.copy(current_EpsY))
        #F_trial = sym.N(F_trial)

        print('F: {0}'.format(F_trial))

        delF_x_trial = np.array([])
        delF_y_trial = np.array([])
        for ii in range(len(delF_x)):
            delF_x_trial = np.append(delF_x_trial, delF_x[ii](*control_args))
            delF_y_trial = np.append(delF_y_trial, delF_y[ii](*control_args))

        # Make a step, keeping only the real part of the gradient (there is a numerical artifact)
        current_EpsX += gamma * np.real(delF_x_trial)
        current_EpsY += gamma * np.real(delF_y_trial)

        if (N_iter == 10000) & autostop:
            break

        N_iter += 1

    # Convert everything to arrays
    EpsX_list = np.array(EpsX_list)
    EpsY_list = np.array(EpsY_list)
    F_list = np.array(F_list)

    return EpsX_list, EpsY_list, F_list



















#
