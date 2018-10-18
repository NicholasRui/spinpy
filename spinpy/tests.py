from spinpy import objects, simulate
import numpy as np
import matplotlib.pyplot as plt
import qutip as qu
import pdb

def run_dynamics():
    """ Test function which simply runs a single s=0.5 driven spin """
    N=1
    s=0.5

    duration = 0.004

    initial_state = objects.State(N, s, 'up')

    d_list = np.logspace(np.log10(1),np.log10(30000),30)

    for ii in d_list:
        H = objects.Hamiltonian(N, s, include_dipole=False, Ex=1000, omega_d=ii)
        a = simulate_dynamics_me(H, initial_state, duration, steps=200, mode='samestate')

        plt.close()
        if s == 1:
            plt.plot(a.expect[0], label='+1')
            plt.plot(a.expect[1], label='0')
            plt.plot(a.expect[2], label='-1')
        elif s == 0.5:
            plt.plot(a.expect[0], label='+1/2')
            plt.plot(a.expect[1], label='-1/2')
        plt.legend()
        plt.ylim(0,1)
        plt.show()



#testham = objects.Hamiltonian(N, s, include_dipole=False)

#Sx_arr = np.array([])
#Sy_arr = np.array([])
#Sz_arr = np.array([])

#Sx = qu.jmat(s,'x')
#Sy = qu.jmat(s,'y')
#Sz = qu.jmat(s,'z')
#Id = qu.qeye(int(2*s+1))

#for ii in range(N): # ii is the qubit the spin operator is for
#    for jj in range(N): # jj is the qubit you're evaluating the operator at
#        if jj == 0:
#            if ii == jj:
#                Sx_oper = Sx
#                Sy_oper = Sy
#                Sz_oper = Sz
#            else:
#                Sx_oper = Id
#                Sy_oper = Id
#                Sz_oper = Id
#        else:
#            if ii == jj:
#                Sx_oper = qu.tensor(Sx_oper, Sx)
#                Sy_oper = qu.tensor(Sy_oper, Sy)
#                Sz_oper = qu.tensor(Sz_oper, Sz)
#            else:
#                Sx_oper = qu.tensor(Sx_oper, Id)
#                Sy_oper = qu.tensor(Sy_oper, Id)
#                Sz_oper = qu.tensor(Sz_oper, Id)
#    Sx_arr = np.append(Sx_arr, Sx_oper)
#    Sy_arr = np.append(Sy_arr, Sy_oper)
#    Sz_arr = np.append(Sz_arr, Sz_oper)

#Sx_full = np.sum(Sx_arr)
#Sy_full = np.sum(Sy_arr)

#test_unit = 1*qu.basis(3,2)*qu.basis(3,0).dag() + 1*qu.basis(3,1)*qu.basis(3,1).dag() - 1 * qu.basis(3,0)*qu.basis(3,2).dag()
#test_unit = qu.tensor(test_unit, test_unit)

#rez = qu.control.grape_unitary(test_unit, testham, [Sx_full, Sy_full], 30, times=[0, 1, 2, 3, 4, 5])









#
