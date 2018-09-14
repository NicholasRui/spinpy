import spinpy
import numpy as np
import qutip as qu
import pdb

class Hamiltonian:
    """ Spin Hamiltonian object

    == IN ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer) (currently, only s=0.5 and s=1 are supported)
    ** include_dipole, Boolean: if True, include dipole-dipole interaction terms (default: True)
    ** Ex: X laser driving term, MHz (default: 0)
    ** Ey: Y laser driving term, MHz (default: 0)

    == ATTRIBUTES ==
    ** Ex: X laser driving term, MHz
    ** Ey: Y laser driving term, MHz

    == PROPERTIES ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer)
    ** include_dipole, Boolean: if True, Hamiltonian includes dipole-dipole terms
    ** operator, qu.QObj: Hamiltonian operator as a qutip QObj

    == FUTURE ==
    ** Make many of these parameters into kwargs
    ** Implement ability to have spins of different s in same ensemble with dipole-dipole interactions between each other
    ** Ask for a function input which determines whether or not a particle can be driven by laser
    """
    def __init__(self, N, s, include_dipole=True, Ex=0, Ey=0):
        self.__N = N
        self.__include_dipole = include_dipole
        self.__s = s
        self.Ex = Ex
        self.Ey = Ey

        if s * 2 % 1 != 0:
            raise ValueError('s must be either an integer or a half-integer.')
        if s < 0:
            raise ValueError('s cannot be negative.')
        if (s != 0.5) and (s != 1.0):
            raise ValueError('Currently, only s=0.5 and s=1 are supported.')

        # Define constants
        ge = 2.8 # Gyromagnetic ratio of electron, MHz Gauss^-1
        Delta = 2870 # Splitting, MHz (vary this later)
        Bz = 100 # DC z magnetic field, Gauss (vary this later)
        # Keep units consistent in the future

        Delta_arr = Delta * np.ones(N) # Array of splittings (in case we want to vary)
        Bz_arr = Bz * np.ones(N) # Array of Bz

        # Define array of position tuples (in future versions, make this specify-able or something)
        pos_arr = list(zip(np.random.uniform(-10,10,N),np.random.uniform(-10,10,N),np.random.uniform(-10,10,N)))

        # Define spin operators
        Sx = qu.jmat(s,'x')
        Sy = qu.jmat(s,'y')
        Sz = qu.jmat(s,'z')
        Id = qu.qeye(int(2*s+1))

        # Generate lists of the spin operators for specific qubits
        self.__Sx_arr = np.array([])
        self.__Sy_arr = np.array([])
        self.__Sz_arr = np.array([])

        for ii in range(N): # ii is the qubit the spin operator is for
            for jj in range(N): # jj is the qubit you're evaluating the operator at
                if jj == 0:
                    if ii == jj:
                        Sx_oper = Sx
                        Sy_oper = Sy
                        Sz_oper = Sz
                    else:
                        Sx_oper = Id
                        Sy_oper = Id
                        Sz_oper = Id
                else:
                    if ii == jj:
                        Sx_oper = qu.tensor(Sx_oper, Sx)
                        Sy_oper = qu.tensor(Sy_oper, Sy)
                        Sz_oper = qu.tensor(Sz_oper, Sz)
                    else:
                        Sx_oper = qu.tensor(Sx_oper, Id)
                        Sy_oper = qu.tensor(Sy_oper, Id)
                        Sz_oper = qu.tensor(Sz_oper, Id)

            self.__Sx_arr = np.append(self.__Sx_arr, Sx_oper)
            self.__Sy_arr = np.append(self.__Sy_arr, Sy_oper)
            self.__Sz_arr = np.append(self.__Sz_arr, Sz_oper)

        # Include 2 * N spin-1 splitting terms
        if spin == 0.5:
            H = ge * Bz_arr * self.__Sz_arr
        if spin == 1:
            H = Delta_arr * self.__Sz_arr ** 2 + ge * Bz_arr * self.__Sz_arr
        H = np.sum(H)

        # If desired, include dipole-dipole interaction terms
        if include_dipole:
            for ii in range(N):
                for jj in range(ii - 1):
                    pos_i = np.array(pos_arr[ii])
                    pos_j = np.array(pos_arr[jj])
                    r = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    rvec = pos_i - pos_j

                    Si_dot_r = self.__Sx_arr[ii]*rvec[0] + self.__Sy_arr[ii]*rvec[1] + self.__Sz_arr[ii]*rvec[2]
                    Sj_dot_r = self.__Sx_arr[jj]*rvec[0] + self.__Sy_arr[jj]*rvec[1] + self.__Sz_arr[jj]*rvec[2]
                    Si_dot_Sj = self.__Sx_arr[ii]*self.__Sx_arr[jj] + self.__Sy_arr[ii]*self.__Sy_arr[jj] + self.__Sz_arr[ii]*self.__Sz_arr[jj]

                    H += -3 * Si_dot_r * Sj_dot_r / r ** 5 + Si_dot_Sj / r ** 3

        self.__operator = H

    @property
    def N(self):
        return self.__N

    @property
    def s(self):
        return self.__s

    @property
    def include_dipole(self):
        return self.__include_dipole

    @property
    def operator(self):
        # Ex and Ey are two extra terms corresponding to driving a transition
        return self.__operator + self.Ex * np.sum(self.__Sx_arr) + self.Ey * np.sum(self.__Sy_arr)

def get_spin_operators_ensemble(ensemble_dict):
    """ A function to produce the full spin operators for an ensemble of spins which may or may not
    all have the same s.

    == IN ==
    ensemble_dict, dict: Dictionary which is structured like {s1: N1, s2: N2, ...} where Ni is the number
           of spins with s=si.

    == OUT ==
    Sx_arr, np.ndarray: Array of full Sx operators for each spin
    Sy_arr, np.ndarray: Array of full Sy operators for each spin
    Sz_arr, np.ndarray: Array of full Sz operators for each spin
    """
    # Get sorted (from highest to lowest) list of s values present in this ensemble
    spins = list(ensemble_dict.keys())
    spins.sort()
    spins = np.flipud(spins)

    # Get total number of spins and an array encoding the s value of the ith spin:
    N = 0
    spin_value = np.array([])
    for s in spins:
        N += ensemble_dict[s]
        spin_value = np.append(spin_value, s * np.ones(ensemble_dict[s]))

        if s * 2 % 1 != 0:
            raise ValueError('s must be either an integer or a half-integer.')
        if s < 0:
            raise ValueError('s cannot be negative.')

    # Now produce an array of Sx, Sy, and Sz operators with the highest spins sorted
    Sx_arr = np.array([])
    Sy_arr = np.array([])
    Sz_arr = np.array([])

    for ii in range(N): # ii is the qubit the spin operator is for
        for jj in range(N): # jj is the qubit you're evaluating the operator at
            if jj == 0:
                if ii == jj:
                    Sx_oper = qu.jmat(spin_value[jj],'x')
                    Sy_oper = qu.jmat(spin_value[jj],'y')
                    Sz_oper = qu.jmat(spin_value[jj],'z')
                else:
                    Sx_oper = qu.qeye(int(2*spin_value[jj]+1))
                    Sy_oper = qu.qeye(int(2*spin_value[jj]+1))
                    Sz_oper = qu.qeye(int(2*spin_value[jj]+1))
            else:
                if ii == jj:
                    Sx_oper = qu.tensor(Sx_oper, qu.jmat(spin_value[jj],'x'))
                    Sy_oper = qu.tensor(Sy_oper, qu.jmat(spin_value[jj],'y'))
                    Sz_oper = qu.tensor(Sz_oper, qu.jmat(spin_value[jj],'z'))
                else:
                    Sx_oper = qu.tensor(Sx_oper, qu.qeye(int(2*spin_value[jj]+1)))
                    Sy_oper = qu.tensor(Sy_oper, qu.qeye(int(2*spin_value[jj]+1)))
                    Sz_oper = qu.tensor(Sz_oper, qu.qeye(int(2*spin_value[jj]+1)))

        Sx_arr = np.append(Sx_arr, Sx_oper)
        Sy_arr = np.append(Sy_arr, Sy_oper)
        Sz_arr = np.append(Sz_arr, Sz_oper)

    return Sx_arr, Sy_arr, Sz_arr














#
