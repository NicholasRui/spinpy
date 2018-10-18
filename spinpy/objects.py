import spinpy
import sympy as sym
import numpy as np
import qutip as qu
from sympy.physics.quantum import *
import pdb

class Hamiltonian:
    """ Spin Hamiltonian object

    == IN ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer) (currently, only s=0.5 and s=1 are supported)
    ** include_dipole, Boolean: if True, include dipole-dipole interaction terms (default: True)

    == ATTRIBUTES ==
    ** omega_d: Driving frequency, MHz

    == PROPERTIES ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer)
    ** include_dipole, Boolean: if True, Hamiltonian includes dipole-dipole terms
    ** operator, qu.qobj: Hamiltonian operator as a qutip qobj

    == FUTURE ==
    ** Make many of these parameters into kwargs
    ** Implement ability to have spins of different s in same ensemble with dipole-dipole interactions between each other
    ** Ask for a function input which determines whether or not a particle can be driven by laser
    """
    def __init__(self, N, s, include_dipole=True, omega_d=2870):
        self.__N = N
        self.__s = s
        self.__include_dipole = include_dipole
        self.omega_d = omega_d

        if s * 2 % 1 != 0:
            raise ValueError('s must be either an integer or a half-integer.')
        if s < 0:
            raise ValueError('s cannot be negative.')
        if (s != 0.5) and (s != 1.0):
            raise ValueError('Currently, only s=0.5 and s=1 are supported.')

        # Define constants
        ge = 2.8 # Gyromagnetic ratio of electron, MHz Gauss^-1
        Delta = 2870 # Splitting, MHz (vary this later)
        Bz = 2870/2.8 # DC z magnetic field, Gauss (vary this later)
        # Keep units consistent in the future

        Delta_arr = Delta * np.ones(N) # Array of splittings (in case we want to vary)
        Bz_arr = Bz * np.ones(N) # Array of Bz

        # Define array of position tuples (in future versions, make this specify-able or something)
        pos_arr = list(zip(np.random.uniform(-10,10,N),np.random.uniform(-10,10,N),np.random.uniform(-10,10,N)))

        # Generate lists of the spin operators for specific qubits
        self.__Sx_arr = np.array([])
        self.__Sy_arr = np.array([])
        self.__Sz_arr = np.array([])

        for ii in range(N): # ii is the qubit the spin operator is for
            for jj in range(N): # jj is the qubit you're evaluating the operator at
                if jj == 0:
                    if ii == jj:
                        Sx_oper = spin_operator(s, 'x')
                        Sy_oper = spin_operator(s, 'y')
                        Sz_oper = spin_operator(s, 'z')
                    else:
                        Sx_oper = np.identity(int(2*s+1))
                        Sy_oper = np.identity(int(2*s+1))
                        Sz_oper = np.identity(int(2*s+1))
                else:
                    if ii == jj:
                        Sx_oper = np.kron(Sx_oper, Sx)#qu.tensor(Sx_oper, Sx)
                        Sy_oper = np.kron(Sy_oper, Sy)
                        Sz_oper = np.kron(Sz_oper, Sz)
                    else:
                        Sx_oper = np.kron(Sx_oper, Id)
                        Sy_oper = np.kron(Sy_oper, Id)
                        Sz_oper = np.kron(Sz_oper, Id)

            self.__Sx_arr = np.append(self.__Sx_arr, Sx_oper)
            self.__Sy_arr = np.append(self.__Sy_arr, Sy_oper)
            self.__Sz_arr = np.append(self.__Sz_arr, Sz_oper)

        # Include 2 * N spin-1 splitting terms
        if s == 0.5:
            H = ge * Bz_arr * self.__Sz_arr
        if s == 1:
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

    def unitary(self, Ex, Ey, duration):
        # Return the sympy unitary matrix for evolution under this Hamiltonian.
        # Doesn't return the object in qutip-friendly format
        if len(Ex) != len(Ey):
            raise ValueError('Ex and Ey do not have the same length.')

        steps = np.linspace(0,duration,len(Ex) + 1)

        for ii in range(len(Ex)):
            t1 = steps[ii]
            t2 = steps[ii+1]

            # Build the exponential that goes into the time-evolution operator
            # Only the controls hold time-dependence
            int_H_dt = self.__operator * (t2 - t1)
            pdb.set_trace()
            int_H_dt += (Ex[ii] * self.__Sx_arr / self.omega_d) * (np.sin(self.omega_d * t2) - np.sin(self.omega_d * t1))
            int_H_dt += -1 * (Ey[ii] * self.__Sy_arr / self.omega_d) * (np.cos(self.omega_d * t2) - np.cos(self.omega_d * t1))

            # Get time-evolution operator over this single slice
            Uii = (-int_H_dt).expm()

            if ii == 0:
                U = Uii
            else:
                U = U * Uii

            return U

        #return [self.__operator, [np.sum(self.__Sx_arr), Hx_coeff], [np.sum(self.__Sy_arr), Hy_coeff]]
        #return self.__operator + self.Ex * np.sum(self.__Sx_arr) + self.Ey * np.sum(self.__Sy_arr)

    def qutip_operator(self, Ex=0, Ey=0, duration=None):
        # Ex and Ey are two extra terms corresponding to driving a transition
        # Do in the non-interaction frame since there could be multiple splittings in general
        # Duration is only needed if you have arrays for Ex, Ey

        # NOTE NOTE NOTE Put in a user fail-safe for if user tries to not have a duration
        # NOTE NOTE NOTE but wants to have an array Ex, Ey

        if (type(Ex) == float) or (type(Ex) == int):
            Hx_coeff = lambda t, args: Ex * np.cos(self.omega_d * t)
        elif (type(Ex) == list) or (type(Ex) == np.ndarray):
            def Hx_coeff(t, args):
                step = duration / len(Ex)
                if t >= duration:
                    return 0
                else:
                    return Ex[int(np.floor(t / step))] * np.cos(self.omega_d * t)

        if (type(Ey) == float) or (type(Ey) == int):
            Hy_coeff = lambda t, args: Ey * np.sin(self.omega_d * t)
        elif (type(Ey) == list) or (type(Ey) == np.ndarray):
            def Hy_coeff(t, args):
                step = duration / len(Ey)
                if t >= duration:
                    return 0
                else:
                    return Ey[int(np.floor(t / step))] * np.sin(self.omega_d * t)

        return [self.__operator, [np.sum(self.__Sx_arr), Hx_coeff], [np.sum(self.__Sy_arr), Hy_coeff]]
        #return self.__operator + self.Ex * np.sum(self.__Sx_arr) + self.Ey * np.sum(self.__Sy_arr)

def basis(N, val):
    """ Function which returns a basis vector for an N-dimensional Hilbert space with the
    'val'-th basis vector fully populated (index from 0). I.e., if N=4, val=1, return np.array([0,1,0,0]).

    == IN ==
    N, int: Dimension of Hilbert space
    val, int: Basis vector which is fully populated (indexed from 0)
    """
    # Catch bad usage
    if (type(N) != int) or (type(val) != int):
        raise ValueError('Both N and val must be type int.')
    if (val >= N) or (val < 0):
        raise ValueError('Need 0 <= val < N.')

    return np.identity(N)[val]

def spin_operator(s, index):
    """ Function which returns a spin operator with a given spin and m.

    == IN ==
    s, float: Spin quantum number.
    index, str: The spin matrix referenced; can be 'x', 'y', 'z', 'p', or 'm'

    == OUT ==
    mat, np.array: Spin matrix
    """
    # Define spin operators
    m = np.flip(np.arange(-s,s+1,1),0)
    Cp = np.sqrt((s - m) * (s + m + 1))[1:]
    Cm = np.sqrt((s + m) * (s - m + 1))[:-1]

    Sp = np.zeros((int(2*s+1),int(2*s+1)))
    Sm = np.zeros((int(2*s+1),int(2*s+1)))

    # Iterate over each row (J+) or column (J-) and construct
    for ii in range(int(2*s)):
        Sp[ii,ii+1] = Cp[ii]
        Sm[ii+1,ii] = Cm[ii]

    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2j)
    Sz = np.diag(m)

    if index == 'x':
        return Sx
    elif index == 'y':
        return Sy
    elif index == 'z':
        return Sz
    elif index == 'p':
        return Sp
    elif index == 'm':
        return Sm
    else:
        raise ValueError('Invalid spin matrix index specified.')

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
                    Sx_oper = spin_operator(spin_value[jj],'x')
                    Sy_oper = spin_operator(spin_value[jj],'y')
                    Sz_oper = spin_operator(spin_value[jj],'z')
                else:
                    Sx_oper = np.identity(int(2*spin_value[jj]+1))
                    Sy_oper = np.identity(int(2*spin_value[jj]+1))
                    Sz_oper = np.identity(int(2*spin_value[jj]+1))
            else:
                if ii == jj:
                    Sx_oper = np.kron(Sx_oper, spin_operator(spin_value[jj],'x')) #qu.tensor(Sx_oper, qu.jmat(spin_value[jj],'x'))
                    Sy_oper = np.kron(Sy_oper, spin_operator(spin_value[jj],'y'))
                    Sz_oper = np.kron(Sz_oper, spin_operator(spin_value[jj],'z'))
                else:
                    Sx_oper = np.kron(Sx_oper, np.identity(int(2*spin_value[jj]+1))) #qu.tensor(Sx_oper, qu.qeye(int(2*spin_value[jj]+1)))
                    Sy_oper = np.kron(Sy_oper, np.identity(int(2*spin_value[jj]+1)))
                    Sz_oper = np.kron(Sz_oper, np.identity(int(2*spin_value[jj]+1)))

        Sx_arr = np.append(Sx_arr, Sx_oper)
        Sy_arr = np.append(Sy_arr, Sy_oper)
        Sz_arr = np.append(Sz_arr, Sz_oper)

    return Sx_arr, Sy_arr, Sz_arr

class State:
    """ State density matrix object which can be initialized to various states.

    == IN ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer)
    ** kind, str: String encoding kind of state, one of the following:
        > 'up' - all are initialized in spin up (or highest mj state)
        > 'down' - all are initialized in spin up (or lowest mj state)
        > 'mixed' - all spins are maximally mixed

    == PROPERTIES ==
    ** N, int: Number of qubits
    ** s, float: Spin quantum number (integer or half-integer)
    ** kind, str: String encoding kind of state
    ** rho, qu.qobj: Density operator of a spin state as a qutip qobj
    """
    def __init__(self, N, s, kind):
        self.__N = N
        self.__s = s
        self.__kind = kind

        # First, initialize single-qubit density matrix depending on 'kind'
        if s * 2 % 1 != 0:
            raise ValueError('s must be either an integer or a half-integer.')
        if s < 0:
            raise ValueError('s cannot be negative.')

        single_dim = int(2 * s + 1)
        if kind == 'up':
            dens = np.outer(basis(single_dim, 0), basis(single_dim, 0))
        elif kind == 'down':
            dens = np.outer(basis(single_dim, single_dim-1), basis(single_dim, single_dim-1))
        elif kind == 'mixed':
            dens = np.identity(single_dim) / single_dim #qu.maximally_mixed_dm(single_dim)
        else:
            raise ValueError('State kind not recognized.')

        for ii in range(N):
            if ii == 0:
                rho = dens
            else:
                rho = np.kron(rho, dens) #qu.tensor(rho, dens)

        self.__rho = rho

    @property
    def N(self):
        return self.__N

    @property
    def s(self):
        return self.__s

    @property
    def kind(self):
        return self.__kind

    @property
    def rho(self):
        return self.__rho














#
