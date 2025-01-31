#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:27:31 2020

@author: christophe
"""

from math import pi, sin, cos
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
#from matplotlib import rc, rcParams
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#rcParams.update({'font.size': 16})

# Some paramters
_eps =1e-12
_maxiter=500

def _basic_check(A, b, x0):
    """ Common check for clarity """
    n, m = A.shape
    if(n != m):
        raise ValueError("Only square matrix allowed")
    if(b.size != n):
        raise ValueError("Bad rhs size")
    if (x0 is None):
        x0 = np.zeros(n)
    if(x0.size != n):
        raise ValueError("Bad initial value size")
    return x0


def laplace(n):
    """ Construct the 1D laplace operator """
    A = np.zeros((n,n))
    A[0,0] = -2
    A[0,1] = 1
    A[n-1,n-1] = -2
    A[n-1,n-2] = 1
    for i in range(1,n-1):
        A[i,i] = -2
        A[i,i-1] = 1
        A[i,i+1] = 1
    return A


def JOR(A, b, x0=None, omega=0.5, eps=_eps, maxiter=_maxiter):
    """
    Solve A x = b
    - A is the sparse matrix
    - b is the right hand side (np.array)
    - omega is the relaxation parameter
    - eps is the convergence threshold
    - maxiter limits the number of iterations

    Methode itérative stationnaire de sur-relaxation (Jacobi over relaxation)
    Convergence garantie si A est à diagonale dominante stricte
    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est diagonal M = (1./omega) * D


    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals
    """
    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()

    # use spa
    # D = spa.diags(A.diagonal())
    D = np.diag(np.diag(A))
    M = (D/omega)
    N = M - A
    assert np.allclose(A, M - N), "Some error was made in the splitting"

    # inv M with spa (easy since M is diagonal)
    # M_inv = spa.diags(1./M.diagonal())
    # print(M_inv)

    M_inv = np.linalg.inv(M)
    c = M_inv @ b
    M_inv_N = M_inv @ N

    for i in range(maxiter):
        x = M_inv_N @ x + c
        r = b - A @ x
        residual_history.append(np.linalg.norm(r))
        if np.linalg.norm(r) < eps:
            break
    
    return x, residual_history


def SOR(A, b, x0=None, omega=1.5, eps=_eps, maxiter=_maxiter):
    """
    Solve A x = b where
    - A is the sparse matrix
    - b is the right hand side (np.array)
    - omega is the relaxation parameter
    - eps is the convergence threshold
    - maxiter limits the number of iterations

    Methode itérative stationnaire de sur-relaxation successive
    (Successive Over Relaxation)

    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est tri. inf. M = (1./omega) * D - E

    * Divergence garantie pour omega <= 0. ou omega >= 2.0
    * Convergence garantie si A est symétrique définie positive pour
    0 < omega  < 2.
    * Convergence garantie si A est à diagonale dominante stricte pour
    0 < omega  <= 1.

    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals

    """
    if (omega > 2.) or (omega < 0.):
        raise ArithmeticError("SOR will diverge")

    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()

    D = np.diag(np.diag(A))
    E = - np.tril(A, -1)
    F = - np.triu(A, 1)
    assert np.allclose(A, D - E - F), "Some error was made in the splitting 1"

    M = (1/omega) * D - E
    N = M - A
    assert np.allclose(A, M - N), "Some error was made in the splitting 2"

    M_inv = np.linalg.inv(M)
    c = M_inv @ b
    M_inv_N = M_inv @ N

    for i in range(maxiter):
        x = M_inv_N @ x + c
        r = b - A @ x
        residual_history.append(np.linalg.norm(r))
        if np.linalg.norm(r) < eps:
            break
        
    return x, residual_history


def injection(fine, n_inc_H, coarse=None):
    """
    Classical injection, that only keep the value of the coarse nodes
    The modification of coarse is done inplace
    """
    if coarse is None:
        coarse = np.zeros(n_inc_H)
    else:
        coarse[:] = fine[::2]

    return coarse


def interpolation(coarse, n_inc_h, fine=None):
    """
    Classical linear interpolation (the modification of fine is done inplace)
    """

    if fine is None:
        fine = np.zeros(n_inc_h)

    fine[1::2] = coarse
    fine[0] = 0.5 * coarse[0]
    fine[-1] = 0.5 * coarse[-1]
    if n_inc_h >=8:
        fine[2:-2:2] = 0.5 * (coarse[:-1] + coarse[1:])

    return fine


def plot(x, y, custom, label=""):
    """
    A custom plot function, usage:
        f, ax = plot(x, y,'-x', label="u")
    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x, y, custom, label=label);
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"u(x)")
    f.tight_layout()
    return f, ax


def tgcyc(nsegment=64, engine=JOR, function_f=None, **kwargs):
    """
    Two grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing

    Warning: make the good distinction between the number of segments, the
    number of nodes and the number of unknowns
    """

    if (nsegment % 2): raise ValueError("nsegment must be even")
    if function_f is None: # default function
        def function_f(x):
            return np.sin(2.5 * pi * x)

    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    h = 1.0 / nsegment
    H = 2. * h

    n_inc_h = nsegment - 1
    n_inc_H = n_inc_h // 2

    # Full points
    xh = np.linspace(0.,1., n)
    xH = np.linspace(0.,1., n//2 + 1)
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]

    # construction of Laplace operator
    Ah = (1/(h*h)) * laplace(n_inc_h)
    AH = (1/(H*H)) * laplace(n_inc_H)

    # Initial guess 
    u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))
    u = u0.copy()

    # The given right hand side
    bh = function_f(xih)

    # Pre-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)
    rh = bh - Ah @ u

    # Restriction with injection
    rH = injection(rh, n_inc_H)

    # Solve on the coarse grid
    vH = np.linalg.solve(AH, rH)

    # Prolongation
    vh = interpolation(vH, n_inc_h)

    # Update solution
    u += vh

    # Post-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)

    return u


def three_gcyc(nsegment=64, engine=JOR, function_f=None, **kwargs):
    """
    Three grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing

    Warning: make the good distinction between the number of segments, the
    number of nodes and the number of unknowns
    """

    if(nsegment%4): raise ValueError("nsegment must be a multiple of 4")

    if function_f is None: # default function
        def function_f(x):
            return np.sin(2.5 * pi * x)

    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    h = 1.0 / nsegment
    H = 2. * h

    n_inc_h = nsegment - 1
    n_inc_H = n_inc_h // 2

    # Full points
    xh = np.linspace(0.,1., n)
    xH = np.linspace(0.,1., n//2 + 1)
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]

    # construction of Laplace operator
    Ah = (1/(h*h)) * laplace(n_inc_h)
    AH = (1/(H*H)) * laplace(n_inc_H)

    # Initial guess 
    u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))
    u = u0.copy()

    # The given right hand side
    bh = function_f(xih)

    # Pre-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)
    rh = bh - Ah @ u

    # Restriction with injection
    rH = injection(rh, n_inc_H)

    # Solve on the coarse grid using a two grid cycle
    vH = tgcyc(nsegment=nsegment//2, engine=engine, function_f=function_f, **kwargs)

    # Prolongation
    vh = interpolation(vH, n_inc_h)

    # Update solution
    u += vh

    # Post-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)

    return u


def multi_gcyc(nb_grid=2, nsegment=64, engine=JOR, function_f=None, **kwargs):
    """
    Multi grid cycle:
        - nb_grid: the number of grid cycle
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing

    Warning: make the good distinction between the number of segments, the
    number of nodes and the number of unknowns
    """

    if(nsegment%(2**(nb_grid-1))): raise ValueError("nsegment must be a multiple of 2^(nb_grid-1)")
    if nsegment <= 2**nb_grid: raise ValueError("nsegment must be greater than 2^nb_grid")

    if function_f is None: # default function
        def function_f(x):
            return np.sin(2.5 * pi * x)

    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    h = 1.0 / nsegment
    H = 2. * h

    n_inc_h = nsegment - 1
    n_inc_H = n_inc_h // 2

    # Full points
    xh = np.linspace(0.,1., n)
    xH = np.linspace(0.,1., n//2 + 1)
    # Inner points
    xih = xh[1:-1]
    xiH = xH[1:-1]

    # construction of Laplace operator
    Ah = (1/(h*h)) * laplace(n_inc_h)
    AH = (1/(H*H)) * laplace(n_inc_H)

    # Initial guess 
    u0 = 0.5 * (np.sin(16. * xih * pi) + np.sin(40. * xih * pi))
    u = u0.copy()

    # The given right hand side
    bh = function_f(xih)

    # Pre-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)
    rh = bh - Ah @ u

    # Restriction with injection
    rH = injection(rh, n_inc_H)

    # Solve on the coarse grid using a multi grid cycle (recursive)
    if nb_grid == 2:
        vH = np.linalg.solve(AH, rH)
    else:
        vH = multi_gcyc(nb_grid-1, nsegment//2, engine=engine, function_f=function_f, **kwargs)

    # Prolongation
    vh = interpolation(vH, n_inc_h)

    # Update solution
    u += vh

    # Post-smoothing Relaxation
    u, _ = engine(Ah, bh, u, **kwargs)

    return u


def plot_multi_gcyc(list_nb_grid, function_f):
    fig, axs = plt.subplots(len(list_nb_grid), 2, figsize=(10, 10))
    for i, nb_grid in enumerate(list_nb_grid):
        list_n_segment = [2**j for j in range(nb_grid+1, 8)]
        nb_colors = len(list_n_segment)
        colors = plt.cm.viridis(np.linspace(0, 1, nb_colors))
        for j, nsegment in enumerate(list_n_segment):
            h = 1.0 / nsegment
            Ah = (1/(h*h)) * laplace(nsegment-1)
            u_jor = multi_gcyc(nb_grid=nb_grid, nsegment=nsegment, engine=JOR, function_f=function_f)
            u_sor = multi_gcyc(nb_grid=nb_grid, nsegment=nsegment, engine=SOR, function_f=function_f)
            x = np.linspace(0.,1., nsegment-1)
            axs[i, 0].plot(x, Ah @ u_jor, label=f"JOR {nsegment}", linestyle="--", c=colors[j], marker="+")
            axs[i, 1].plot(x, Ah @ u_sor, label=f"SOR {nsegment}", linestyle="--", c=colors[j], marker="+")
        x = np.linspace(0.,1., 100)
        axs[i, 0].plot(x, function_f(x), label="f", linestyle="-", c="r")
        axs[i, 1].plot(x, function_f(x), label="f", linestyle="-", c="r")
        axs[i, 0].set_title(f"{list_nb_grid[i]} grid cycle - JOR")
        axs[i, 1].set_title(f"{list_nb_grid[i]} grid cycle - SOR")
        axs[i, 0].legend()
        axs[i, 1].legend()
    plt.tight_layout()
    plt.show()


# For debugging:
tgcyc(nsegment=8)

# For real application
# tgcyc(nsegment=64)