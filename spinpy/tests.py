from spinpy import objects as ob
from spinpy import simulate as sim
import numpy as np
import matplotlib.pyplot as plt
import qutip as qu
import pdb

def GRAPE_single_spin_flip(N=3, gamma=0.1):
    """ Test function which attempts to find the optimal pulse to flip a single spin """
    #N=3
    s=0.5

    H = ob.Hamiltonian(1, s, include_dipole=True, omega_d=2.8)
    res = sim.GRAPE_pulse(H, 3, N, gamma=gamma, thres=1e-8)
    #res = sim.GRAPE_pulse(H, 3, 1, gamma=gamma, thres=1e-8)

    return res

def determined_initial_state(N=1, gamma=0.1):
    n=2
    s=0.5
    kind='up'
    gamma = 0.01

    H = ob.Hamiltonian(n, s, include_dipole=True, omega_d=2.8)
    init_state = ob.State(n, s, kind)
    res = sim.GRAPE_pulse(H, 1, N, gamma=gamma, thres=1e-8, init_state=init_state)

    return res

def B_disorder_test(N=1, s=0.5, n=1, Brms=1):
    """ Code which introduces some magnetic noise with some RMS """
    s=0.5
    kind='up'
    gamma = 0.01

    H = ob.Hamiltonian(n, s, include_dipole=False, omega_d=2.8, Brms=Brms)
    init_state = ob.State(n, s, kind)
    res = sim.GRAPE_pulse(H, 1, N, gamma=gamma, thres=1e-8, init_state=init_state)

    return res

def B_disorder_investigate(Brms):
    """ Code which takes in a magnetic field RMS Brms and returns a list of 50
    output fidelities """
    n=1
    s=0.5
    kind='up'
    gamma = 0.01
    init_state = ob.State(n, s, kind)
    N = 1
    F_list = []

    for ii in range(50):
        H = ob.Hamiltonian(n, s, include_dipole=False, omega_d=2.87, Brms=Brms)
        EpsX, EpsY, F = sim.GRAPE_pulse(H, 1, N, gamma=gamma, thres=1e-8, init_state=init_state, autostop=True)
        F_list.append(F[-1])

    F_list = np.array(F_list)
    np.save('Brms_{0}_.npy'.format(Brms), F_list)

def do_my_bidding():
    """ A function to run tests.B_disorder_investigate many times for many Brms values """
    Brms_list = [0.01, 0.03, 0.1, 0.3, 1, 3]

    for B in Brms_list:
        B_disorder_investigate(B)

def plot_rms_results():
    """ A function to plot the results of do_my_bidding() """
    fnames = ['Brms_0.01_.npy',
        'Brms_0.03_.npy',
        'Brms_0.1_.npy',
        'Brms_0.3_.npy',
        'Brms_1_.npy',
        'Brms_3_.npy']

    rms = ['0.01 G','0.03 G','0.10 G',
           '0.30 G','1.00 G','3.00 G']

    bins = np.logspace(-9,0,10)

    plt.close()
    for ii in range(len(fnames)):
        data = np.load(fnames[ii])
        hist, bin_edges = np.histogram(1-data, bins)
        bin_center = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        plt.plot(bin_center, hist, label=rms[ii])

    plt.legend(fontsize=14)
    plt.xscale('log')
    plt.xlabel(r'$1-F$', fontsize=20)
    plt.ylabel(r'Count', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('fidel.png', bbox_inches='tight')


def ordered_integral_test(N, s):
    for ii in range(N): # ii is the qubit the spin operator is for
        for jj in range(N): # jj is the qubit you're evaluating the operator at
            if jj == 0:
                if ii == jj:
                    Sx_oper = spin_operator(s, 'x')
                    Sy_oper = spin_operator(s, 'y')
                    Sz_oper = spin_operator(s, 'z')
                else:
                    Sx_oper = sym.eye(int(2*s+1))
                    Sy_oper = sym.eye(int(2*s+1))
                    Sz_oper = sym.eye(int(2*s+1))
            else:
                if ii == jj:
                    Sx_oper = TensorProduct(Sx_oper, spin_operator(s, 'x'))
                    Sy_oper = TensorProduct(Sy_oper, spin_operator(s, 'y'))
                    Sz_oper = TensorProduct(Sz_oper, spin_operator(s, 'z'))
                else:
                    Sx_oper = TensorProduct(Sx_oper, sym.eye(int(2*s+1)))
                    Sy_oper = TensorProduct(Sy_oper, sym.eye(int(2*s+1)))
                    Sz_oper = TensorProduct(Sz_oper, sym.eye(int(2*s+1)))

        self.__Sx_arr.append(Sx_oper)
        self.__Sy_arr.append(Sy_oper)
        self.__Sz_arr.append(Sz_oper)

    # Get some random dipole terms or whatever
    dipole_term = None
    m = sym.symbols('m')
    pos_arr = list(zip(np.random.uniform(-10,10,N),np.random.uniform(-10,10,N),np.random.uniform(-10,10,N)))
    for ii in range(N):
        for jj in range(ii):
            pos_i = np.array(pos_arr[ii])
            pos_j = np.array(pos_arr[jj])
            r = np.sqrt(np.sum((pos_i - pos_j) ** 2))
            rvec = pos_i - pos_j

            Si_dot_r = self.__Sx_arr[ii]*rvec[0] + self.__Sy_arr[ii]*rvec[1] + self.__Sz_arr[ii]*rvec[2]
            Sj_dot_r = self.__Sx_arr[jj]*rvec[0] + self.__Sy_arr[jj]*rvec[1] + self.__Sz_arr[jj]*rvec[2]
            Si_dot_Sj = self.__Sx_arr[ii]*self.__Sx_arr[jj] + self.__Sy_arr[ii]*self.__Sy_arr[jj] + self.__Sz_arr[ii]*self.__Sz_arr[jj]

            if dipole_term == None:
                dipole_term = -3 * m * Si_dot_r * Sj_dot_r / r ** 5 + m * Si_dot_Sj / r ** 3
            else:
                dipole_term += -3 * m * Si_dot_r * Sj_dot_r / r ** 5 + m * Si_dot_Sj / r ** 3

            pdb.set_trace()

    print('=============')
    print('Dipole term')
    print('=============')
    print(dipole_term)

    # Get the driving terms and stuff

#def compare_H():
#    """ See what the effect is of adding magnetic noise terms """
#    H = ob.Hamiltonian(n, s, include_dipole=True, omega_d=2.8, Brms=20)







#
