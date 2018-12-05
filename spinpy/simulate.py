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

# This function is broken. Do not use.
def num_GRAPE_pulse(H, duration, N, thres=1e-2):#initmag=10):
    """ Main function to optimize pulse with GRAPE.
    ** H, qu.qobj: Hamiltonian operator as a qutip qobj
    ** duration, float: Time length of the pulse
    ** N, int: Number of time slices to split Ex and Ey into
    ** gamma, float: Learning parameter, at each step, simulator jumps by gamma * del f
    ** thres, float: Tolerance threshold for final converged run.

    == OUT ==

    """
    ################################################
    # Setup
    ################################################

    # Construct basis vectors and collapse operators for Hilbert space
    b_vec = []
    e_ops = []

    for ii in range(int(2*H.s+1)):
        basis_state = ob.basis(int(2*H.s+1),ii)#qu.basis(int(2*H.s+1), ii)
        for jj in range(H.N):
            if jj == 0:
                basis_vec = basis_state
                collapse_op = np.outer(basis_state, basis_state)#basis_state * basis_state.dag()
            else:
                basis_vec = np.kron(basis_vec, basis_state)#qu.tensor(basis_vec, basis_state)
                collapse_op = np.kron(collapse_op, np.outer(basis_state, basis_state))#qu.tensor(collapse_op, basis_state * basis_state.dag())
        b_vec.append(basis_vec)
        e_ops.append(collapse_op)

    # Define options object to increase number of steps allowed
    #options = qu.Options(nsteps=1e9) #some large number

    ################################################
    # Build desired unitary
    ################################################

    # Construct the desired unitary. For now, pick the one that swaps all of them down
    for ii in range(H.N):
        if ii == 0:
            Uideal = ob.spin_operator(0.5,'x')#qu.sigmax()
        else:
            Uideal = np.kron(Uideal, ob.spin_operator(0.5,'x'))#qu.tensor(Uideal, qu.sigmax())
    #Uideal = np.sum(-1 * e_ops[0] + np.sum([e_ops[ii+2] for ii in range(len(e_ops) - 2)]))

    # Construct the ideal bra vectors
    bras_ideal = [np.conj(np.transpose(Uideal * b_vec)) for ii in range(len(b_vec))] #[b_vec[ii].dag() * Uideal.dag() for ii in range(len(b_vec))]

    ################################################
    # Initialize guess and simulation arrays
    ################################################

    # Dumb initial guess, uniform driving with no normalization whatsoever
    current_EpsX = 4 * np.pi * omega_d * np.ones(N) / (ge * Bz * duration)#initmag * np.ones(N)
    current_EpsY = np.zeros(N)#0 * initmag * np.ones(N)

    N_iter = 1
    phi_list = [0] # Initialize list of cost-functions (with 0 starting it off)
    EpsX_list = []
    EpsY_list = []

    # Define machine epsilon (for stepping)
    eps = 1e-6#*np.finfo(float).eps#0.001#

    ################################################
    # The fun bit
    ################################################
    while (1 - phi_list[-1]) > thres:
        start = time.time()

        # Initialize list of partials with respect to Sx and Sy quadrature terms
        dx_arr = np.array([])
        dy_arr = np.array([])

        phi = 0

        for jj in range(len(b_vec)): # Iterate over basis vectors to find Ugrape|i>
            # Record current EpsX, EpsY
            EpsX_list.append(current_EpsX)
            EpsY_list.append(current_EpsY)

            # Evolve each individual basis vector
            q = qu.mesolve(H.operator(current_EpsX, current_EpsY, duration), b_vec[jj],
                           tlist=np.array([0, duration]), options=options)

            # Check how close the output basis vector is to the desired state
            final_state = q.states[-1]
            phi += (bras_ideal[jj] * final_state)[0][0][0]

        # Calculate generalized version of Eqn. 11 in Motzoi+2009 (cost function)
        phi = np.abs(phi) ** 2 / len(b_vec) ** 2
        phi_list.append(phi)
        print('Iteration: {0}, PHI = {1}'.format(N_iter,phi))

        for ii in range(len(current_EpsX)): # Perturb each Ex element to compute numerical gradient
            phi_ds = 0
            EpsX_test = current_EpsX
            EpsX_test[ii] += eps

            for jj in range(len(b_vec)):
                q = qu.mesolve(H.operator(EpsX_test, current_EpsY, duration), b_vec[jj],
                               tlist=np.array([0, duration]), options=options)

                final_state = q.states[-1]
                phi_ds += (bras_ideal[jj] * final_state)[0][0][0]

            phi_ds = np.abs(phi_ds) ** 2 / len(b_vec) ** 2
            dx_arr = np.append(dx_arr, (phi_ds - phi) / eps)

        print('           X done'.format(N_iter,phi))

        for ii in range(len(current_EpsY)): # Perturb each Ey element to compute numerical gradient
            phi_ds = 0
            EpsY_test = current_EpsY
            EpsY_test[ii] += eps

            for jj in range(len(b_vec)):
                q = qu.mesolve(H.operator(current_EpsX, EpsY_test, duration), b_vec[jj],
                               tlist=np.array([0, duration]), options=options)

                final_state = q.states[-1]
                phi_ds += (bras_ideal[jj] * final_state)[0][0][0]

            phi_ds = np.abs(phi_ds) ** 2 / len(b_vec) ** 2
            dy_arr = np.append(dy_arr, (phi_ds - phi) / eps)

        print('           Y done'.format(N_iter,phi))
        print('           Iter time: {0} s'.format(time.time() - start))

        # Update guesses based on gradient
        # NOTE: Should make sure this proportionality constant makes sense
        current_EpsX += gamma * dx_arr
        current_EpsY += gamma * dy_arr

        print(current_EpsX)
        print(current_EpsY)

        N_iter += 1

        if N_iter == 100:
            pdb.set_trace()

    return current_EpsX, current_EpsY

def GRAPE_pulse(H, duration, N, gamma=0.01, thres=1e-8):
    """ Main function to optimize pulse with GRAPE.
    ** H, qu.qobj: Hamiltonian operator as a qutip qobj
    ** duration, float: Time length of the pulse
    ** N, int: Number of time slices to split Ex and Ey into
    ** gamma, float: Learning parameter, at each step, simulator jumps by gamma * del f
    ** thres, float: Tolerance threshold for final converged run.

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
    U_lambda = sym.lambdify(np.append(X_control, Y_control), U)

    # Calculate the analytical gradient in both quadratures
    #delU_x = np.array([])
    #delU_y = np.array([])
    #
    #for ii in range(N):
    #    delU_x = np.append(delU_x, sym.diff(U, X_control[ii]))
    #    delU_y = np.append(delU_y, sym.diff(U, Y_control[ii]))

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
    F = 0
    for ii in range(len(b_vec)):
        # Calculate <n|Uideal * Ugrape|n>
        term = (bras_ideal[ii] * U * b_vec[ii])[0,0]
        print(ii+1)
        F += term

    # Get fidelity, convert to fast lambda function
    F = sym.conjugate(F) * F / len(b_vec)**2
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
    current_EpsX = 2 * np.pi * np.ones(N) / duration#(np.pi * np.ones(N) / duration)
    current_EpsY = np.zeros(N)

    # Look at initial Uideal to see if it's reasonable
    Uinit = sym.Matrix(U_lambda(*current_EpsX, *current_EpsY))

    N_iter = 1
    F_list = [0] # Initialize list of fidelities
    EpsX_list = []
    EpsY_list = []

    # Define machine epsilon (for stepping)
    #eps = 1e-6#*np.finfo(float).eps#0.001#

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
        control_args = np.append(current_EpsX, current_EpsY)

        #param_dict = {}
        #param_dict = {}
        #for ii in range(len(X_control)):
        #    param_dict[X_control[ii]] = current_EpsX[ii]
        #    param_dict[Y_control[ii]] = current_EpsY[ii]

        # Calculate trial fidelity. Take real fidelity
        F_trial = np.real(F_func(*control_args))#F.xreplace(param_dict)

        # Append everything to arrays
        F_list.append(F_trial)
        EpsX_list.append(np.copy(current_EpsX))
        EpsY_list.append(np.copy(current_EpsY))
        #F_trial = sym.N(F_trial)

        print('F: {0}'.format(F_trial))

        delF_x_trial = np.array([])
        delF_y_trial = np.array([])
        for ii in range(len(delF_x)):
            delF_x_trial = np.append(delF_x_trial, delF_x[ii](*control_args))#delF_x[ii].xreplace(param_dict))
            delF_y_trial = np.append(delF_y_trial, delF_y[ii](*control_args))#delF_y[ii].xreplace(param_dict))

        #if N_iter % 100 == 0:
        #    pdb.set_trace()

        # Make a step, keeping only the real part of the gradient (there is a numerical artifact)
        current_EpsX += gamma * np.real(delF_x_trial)
        current_EpsY += gamma * np.real(delF_y_trial)

        N_iter += 1

    # Convert everything to arrays
    EpsX_list = np.array(EpsX_list)
    EpsY_list = np.array(EpsY_list)
    F_list = np.array(F_list)

    return EpsX_list, EpsY_list, F_list



















#
