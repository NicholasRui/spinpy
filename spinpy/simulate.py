from spinpy import objects
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
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

def num_GRAPE_pulse(H, duration, N, gamma=3000, thres=1e-2, initmag=10):
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
        basis_state = qu.basis(int(2*H.s+1), ii)
        for jj in range(H.N):
            if jj == 0:
                basis_vec = basis_state
                collapse_op = basis_state * basis_state.dag()
            else:
                basis_vec = qu.tensor(basis_vec, basis_state)
                collapse_op = qu.tensor(collapse_op, basis_state * basis_state.dag())
        b_vec.append(basis_vec)
        e_ops.append(collapse_op)

    # Define options object to increase number of steps allowed
    options = qu.Options(nsteps=1e9) #some large number

    ################################################
    # Build desired unitary
    ################################################

    # Construct the desired unitary. For now, pick the one that swaps all of them down
    for ii in range(H.N):
        if ii == 0:
            Uideal = qu.sigmax()
        else:
            Uideal = qu.tensor(Uideal, qu.sigmax())
    #Uideal = np.sum(-1 * e_ops[0] + np.sum([e_ops[ii+2] for ii in range(len(e_ops) - 2)]))

    # Construct the ideal bra vectors
    bras_ideal = [b_vec[ii].dag() * Uideal.dag() for ii in range(len(b_vec))]

    ################################################
    # Initialize guess and simulation arrays
    ################################################

    # Dumb initial guess, uniform driving with no normalization whatsoever
    current_EpsX = initmag * np.ones(N)
    current_EpsY = 0 * initmag * np.ones(N)

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

def GRAPE_pulse(H, duration, N, gamma=3000, thres=1e-2, initmag=10):
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
        basis_state = qu.basis(int(2*H.s+1), ii)
        for jj in range(H.N):
            if jj == 0:
                basis_vec = basis_state
                collapse_op = basis_state * basis_state.dag()
            else:
                basis_vec = qu.tensor(basis_vec, basis_state)
                collapse_op = qu.tensor(collapse_op, basis_state * basis_state.dag())
        b_vec.append(basis_vec)
        e_ops.append(collapse_op)

    X_control = np.array(sym.symbols('x0:{0}'.format(N)))
    Y_control = np.array(sym.symbols('x0:{0}'.format(N)))

    H.operator(X_control, Y_control)

    # Construct the ideal bra vectors
    bras_ideal = [b_vec[ii].dag() * Uideal.dag() for ii in range(len(b_vec))]

    ################################################
    # Initialize guess and simulation arrays
    ################################################

    # Dumb initial guess, uniform driving with no normalization whatsoever
    current_EpsX = initmag * np.ones(N)
    current_EpsY = 0 * initmag * np.ones(N)

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



















#
