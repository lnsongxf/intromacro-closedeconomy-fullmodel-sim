#!/usr/bin/env python3
# Preamble:{{{1
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

import os
import sys

# Get full real filename
if os.path.islink(__file__):
    __fullrealfile__ = os.readlink(__file__)
else:
    __fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = getprojectdir(__fullrealfile__)

# Setting up importattr (function to load attributes from files by absolute path)
if 'fullmodulesloaded' not in globals():
    fullmodulesloaded = {}
sys.path.append(__projectdir__ + 'submodules/python-preamble/')
from importattr2 import importattr2

def importattr(filename, function, fullmodulesloaded = fullmodulesloaded, curfilename = __fullrealfile__, tryimport = True):
    if os.path.abspath(filename) == curfilename:
        func = eval(function)
    else:
        func = importattr2(filename, function, fullmodulesloaded, tryimport = tryimport)
    return(func)

# PYTHON_PREAMBLE_END:}}}


import matplotlib.pyplot as plt
import functools
import numpy as np
from scipy.optimize import fsolve
import shutil
import warnings

# General Definitions:{{{1
ALPHA = 0.6
modelfiguretex = r"""\begin{figure}[H]
    \centering
    \begin{minipage}{.49\textwidth}
        \centering
        \includegraphics[width=0.9\linewidth]{graph1.jpg}
    \end{minipage}
    \begin{minipage}{.49\textwidth}
        \centering
        \includegraphics[width=0.9\linewidth]{graph2.jpg}
    \end{minipage}
    \begin{minipage}{.49\textwidth}
        \centering
        \includegraphics[width=0.9\linewidth]{graph3.jpg}
    \end{minipage}
    \begin{minipage}{.49\textwidth}
        \centering
        \includegraphics[width=0.9\linewidth]{graph4.jpg}
    \end{minipage}
    \begin{minipage}{.49\textwidth}
        \centering
        \includegraphics[width=0.9\linewidth]{graph5.jpg}
    \end{minipage}
    \caption{modelname}
\end{figure}"""

# Defaults:{{{1
def defaultsdictgen():
    defaults = {}
    defaults['T'] = 1
    defaults['G'] = 1
    defaults['M'] = 1
    defaults['K'] = 8

    defaults['Wbar'] = 1
    defaults['WPbar'] = 1.5
    return(defaults)


def multipleofdefaults(multipledict):
    """
    Specify K = 1.01 in multipledict to mean that K is the default * 1.01
    """
    newdict = {}
    defaultdict = defaultsdictgen()
    for item in defaultdict:
        if item in multipledict:
            newdict[item] = multipledict[item] * defaultdict[item]
        else:
            newdict[item] = defaultdict[item]

    return(newdict)
            
            
# Equations:{{{1
def is_equation(G, i, T, Y):
    return(0.8 * (Y-T) + 0.1 * (Y-i) + G - Y)

def lm_equation(i, M, P, Y):
    return(M - P * Y * (1 - i))

def production_equation(K, N, Y):
    return(Y - K ** ALPHA * N * (1 - ALPHA))

def laboursupply_equation(N, P, W):
    return(N - W / float(P))

def labourdemand_equation(K, N, P, W):
    return(W/ float(P) - (1 - ALPHA) * K** ALPHA * N ** (- ALPHA))


def stickyrealwage_equation(P, WP, W):
    return(WP - W / float(P))


# Equilibria:{{{1
def classicalequilibrium_tosolve(x, exog):
    """
    Equation I solve to get equilibrium in classical case.
    """
    Y = x[0]
    P = x[1]
    i = x[2]
    N = x[3]
    W = x[4]
    
    eqsolve = []
    eqsolve.append(is_equation(exog['G'], i, exog['T'], Y))
    eqsolve.append(lm_equation(i, exog['M'], P, Y))
    eqsolve.append(production_equation(exog['K'], N, Y))
    eqsolve.append(laboursupply_equation(N, P, W))
    eqsolve.append(labourdemand_equation(exog['K'], N, P, W))

    return(eqsolve)


def classicalequilibrium(values):

    f = functools.partial(classicalequilibrium_tosolve, exog = values)

    sol = fsolve(f, [1] * 5)

    values['Y'] = sol[0]
    values['P'] = sol[1]
    values['i'] = sol[2]
    values['N'] = sol[3]
    values['W'] = sol[4]


    return(values)


def stickynominalwageequilibrium_tosolve(x, exog):
    """
    Equation I solve to get equilibrium in classical case.
    """
    Y = x[0]
    P = x[1]
    i = x[2]
    N = x[3]
    
    eqsolve = []
    eqsolve.append(is_equation(exog['G'], i, exog['T'], Y))
    eqsolve.append(lm_equation(i, exog['M'], P, Y))
    eqsolve.append(production_equation(exog['K'], N, Y))
    eqsolve.append(labourdemand_equation(exog['K'], N, P, exog['Wbar']))

    return(eqsolve)


def stickynominalwageequilibrium(values):

    f = functools.partial(stickynominalwageequilibrium_tosolve, exog = values)

    sol = fsolve(f, [1] * 4)

    values['Y'] = sol[0]
    values['P'] = sol[1]
    values['i'] = sol[2]
    values['N'] = sol[3]

    values['W'] = values['Wbar']


    return(values)


def stickyrealwageequilibrium_tosolve(x, exog):
    """
    Equation I solve to get equilibrium in classical case.
    """
    Y = x[0]
    P = x[1]
    i = x[2]
    N = x[3]
    W = x[4]
    
    eqsolve = []
    eqsolve.append(is_equation(exog['G'], i, exog['T'], Y))
    eqsolve.append(lm_equation(i, exog['M'], P, Y))
    eqsolve.append(production_equation(exog['K'], N, Y))
    eqsolve.append(laboursupply_equation(N, P, W))
    eqsolve.append(stickyrealwage_equation(P, exog['WPbar'], W))

    return(eqsolve)


def stickyrealwageequilibrium(values):

    f = functools.partial(stickyrealwageequilibrium_tosolve, exog = values)

    sol = fsolve(f, [1] * 5)

    values['Y'] = sol[0]
    values['P'] = sol[1]
    values['i'] = sol[2]
    values['N'] = sol[3]
    values['W'] = sol[4]


    return(values)


# Single Graphs:{{{1
def asad_graph_ad(values):
    Yvec = np.linspace(0.4 * values['Y'], 1.6 * values['Y'])

    Pvec = []
    for Y in Yvec:
        def f(P):
            return(lm_equation(values['i'], values['M'], P, Y))

        sol = fsolve(f, values['P'])[0]
        Pvec.append(sol)

    return(Yvec, Pvec)


def asad_graph_as(values):
    Pvec = np.linspace(0.4 * values['P'], 1.6 * values['P'])

    Yvec = []
    for P in Pvec:
        # first solve for N
        def f(N):
            return(labourdemand_equation(values['K'], N, P, values['W']))

        # turn off/on warnings to get rid of message about invalid power
        warnings.simplefilter('ignore')
        N, infodict, ier, mesg = fsolve(f, values['N'], full_output = True)
        warnings.simplefilter('default')

        if ier == 1:
            N = N[0]
        else:
            roots = importattr(__projectdir__ + 'submodules/notme-short/python_roots.py', 'roots2')(f, values['N'] * 0.01, values['N'] * 3, 1e-3)
            if len(roots) == 1:
                N = list(roots)[0]


        # input value of N into Y
        def f(Y):
            return(production_equation(values['K'], N, Y))

        sol = fsolve(f, values['P'])[0]

        Yvec.append(sol)

    return(Yvec, Pvec)


def asad_graph(before, after = None, savename = None, savedir = None):
    """
    Only use for sticky nominal wage.
    Misleading to use for classical model since changing P does actually change Y in classical.
    """

    if savedir is not None:
        savename = savedir + 'asad.jpg'

    Yvec, Pvec = asad_graph_ad(before)
    plt.plot(Yvec, Pvec, label = r'$AD_0$')

    Yvec, Pvec = asad_graph_as(before)
    plt.plot(Yvec, Pvec, label = r'$AS_0$')

    if after is not None:
        Yvec, Pvec = asad_graph_ad(after)
        plt.plot(Yvec, Pvec, label = r'$AD_1$')

        Yvec, Pvec = asad_graph_as(after)
        plt.plot(Yvec, Pvec, label = r'$AS_1$')


    plt.title('AS-AD')
    plt.xlabel(r'Output ($Y$)')
    plt.ylabel(r'Price Level ($P$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()

    
def islm_graph_is(values):
    """
    Find i in terms of other variables
    """
    Yvec = np.linspace(0.4 * values['Y'], 1.6 * values['Y'])

    ivec = []
    for Y in Yvec:
        def f(i):
            return(is_equation(values['G'], i, values['T'], Y))

        sol = fsolve(f, values['i'])[0]
        ivec.append(sol)

    return(Yvec, ivec)


def islm_graph_lm(values):
    """
    Find i in terms of other variables
    """
    Yvec = np.linspace(0.4 * values['Y'], 1.6 * values['Y'])

    ivec = []
    for Y in Yvec:
        def f(i):
            return(lm_equation(i, values['M'], values['P'], Y))

        sol = fsolve(f, values['i'])[0]
        ivec.append(sol)

    return(Yvec, ivec)


def islm_graph(before, after = None, savename = None, savedir = None):

    if savedir is not None:
        savename = savedir + 'islm.jpg'

    Yvec, ivec = islm_graph_is(before)
    plt.plot(Yvec, ivec, label = r'$IS_0$')

    Yvec, ivec = islm_graph_lm(before)
    plt.plot(Yvec, ivec, label = r'$LM_0$')

    if after is not None:
        Yvec, ivec = islm_graph_is(after)
        plt.plot(Yvec, ivec, label = r'$IS_1$')

        Yvec, ivec = islm_graph_lm(after)
        plt.plot(Yvec, ivec, label = r'$LM_1$')

    plt.title('IS-LM')
    plt.xlabel(r'Output ($Y$)')
    plt.ylabel(r'Interest Rate ($i$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()

    
def money_graph_demand(values):
    """
    Find i in terms of other variables
    """
    MP = values['M'] / float(values['P'])
    MPvec = np.linspace(0.4 * MP, 1.6 * MP)

    ivec = []
    for MP in MPvec:
        def f(i):
            return(lm_equation(i, MP, 1, values['Y']))

        sol = fsolve(f, values['i'])[0]
        ivec.append(sol)

    return(MPvec, ivec)


def money_graph_supply(values, ivec):
    """
    Find i in terms of other variables
    """
    mini = min(0, min(ivec))
    maxi = max(0, max(ivec))
    ivec = np.linspace(mini, maxi)

    MPvec = []
    for i in ivec:
        MPvec.append(values['M'] / float(values['P']))

    return(MPvec, ivec)


def money_graph(before, after = None, savename = None, savedir = None):

    if savedir is not None:
        savename = savedir + 'money.jpg'

    MPvec, ivec = money_graph_demand(before)
    plt.plot(MPvec, ivec, label = r'Demand 0 ($YL(i)$)')

    MPvec, ivec = money_graph_supply(before, ivec)
    plt.plot(MPvec, ivec, label = r'Supply 0 ($\frac{M}{P}$)')

    if after is not None:
        MPvec, ivec = money_graph_demand(after)
        plt.plot(MPvec, ivec, label = r'Demand 1 ($YL(i)$)')

        MPvec, ivec = money_graph_supply(after, ivec)
        plt.plot(MPvec, ivec, label = r'Supply 1 ($\frac{M}{P}$)')

    plt.title('Money Market')
    plt.xlabel(r'Real Money ($\frac{M}{P}$)')
    plt.ylabel(r'Interest Rate ($i$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])


    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()

def output_graph_output(values):
    """
    Find i in terms of other variables
    """
    Nvec = np.linspace(0.4 * values['N'], 1.6 * values['N'])

    Yvec = []
    for N in Nvec:
        def f(Y):
            return(production_equation(values['K'], N, Y))

        sol = fsolve(f, values['Y'])[0]
        Yvec.append(sol)

    return(Nvec, Yvec)


def output_graph_labour(values, Yvec):
    """
    Find i in terms of other variables
    """
    minY = min(0, min(Yvec))
    maxY = max(0, max(Yvec))
    Yvec = np.linspace(minY, maxY)

    Nvec = []
    for Y in Yvec:
        Nvec.append(values['N'])

    return(Nvec, Yvec)


def output_graph(before, after = None, savename = None, savedir = None):

    if savedir is not None:
        savename = savedir + 'output.jpg'

    Nvec, Yvec = output_graph_output(before)
    plt.plot(Nvec, Yvec, label = r'Output 0 ($Y$)')

    Nvec, Yvec = output_graph_labour(before, Yvec)
    plt.plot(Nvec, Yvec, label = r'Labour 0 ($N$)')

    if after is not None:
        Nvec, Yvec = output_graph_output(after)
        plt.plot(Nvec, Yvec, label = r'Output 1 ($Y$)')

        Nvec, Yvec = output_graph_labour(after, Yvec)
        plt.plot(Nvec, Yvec, label = r'Labour 1 ($N$)')

    plt.title('Production')
    plt.xlabel(r'Labour ($N$)')
    plt.ylabel(r'Output ($Y$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])


    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()


def labour_graph_demand(values):
    Nvec = np.linspace(0.4 * values['N'], 1.6 * values['N'])

    WPvec = []
    for N in Nvec:
        def f(WP):
            return(labourdemand_equation(values['K'], N, 1, WP))

        sol = fsolve(f, values['W'] / float(values['P']))[0]
        WPvec.append(sol)

    return(Nvec, WPvec)


def labour_graph_supply(values):
    Nvec = np.linspace(0.4 * values['N'], 1.6 * values['N'])

    WPvec = []
    for N in Nvec:
        def f(WP):
            return(laboursupply_equation(N, 1, WP))

        sol = fsolve(f, values['W'] / float(values['P']))[0]
        WPvec.append(sol)

    return(Nvec, WPvec)


def labour_graph_stickywage(values):
    Nvec = np.linspace(0.4 * values['N'], 1.6 * values['N'])

    WPvec = []
    for N in Nvec:
        WPvec.append(values['W'] / values['P'])

    return(Nvec, WPvec)


def labourclassical_graph(before, after = None, savename = None, savedir = None):

    if savedir is not None:
        savename = savedir + 'labourclassical.jpg'

    Nvec, WPvec = labour_graph_demand(before)
    plt.plot(Nvec, WPvec, label = r'Labour Demand 0')

    Nvec, WPvec = labour_graph_supply(before)
    plt.plot(Nvec, WPvec, label = r'Labour Supply 0')

    if after is not None:
        Nvec, WPvec = labour_graph_demand(after)
        plt.plot(Nvec, WPvec, label = r'Labour Demand 1')

        Nvec, WPvec = labour_graph_supply(after)
        plt.plot(Nvec, WPvec, label = r'Labour Supply 1')

    plt.title('Labour Market')
    plt.xlabel(r'Labour ($N$)')
    plt.ylabel(r'Real Wage ($\frac{W}{P}$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()

    

def laboursticky_graph(before, after = None, savename = None, savedir = None):

    if savedir is not None:
        savename = savedir + 'laboursticky.jpg'

    Nvec, WPvec = labour_graph_demand(before)
    plt.plot(Nvec, WPvec, label = r'Labour Demand 0')

    Nvec, WPvec = labour_graph_stickywage(before)
    plt.plot(Nvec, WPvec, label = r'Sticky Wage 0')

    if after is not None:
        Nvec, WPvec = labour_graph_demand(after)
        plt.plot(Nvec, WPvec, label = r'Labour Demand 1')

        Nvec, WPvec = labour_graph_stickywage(after)
        plt.plot(Nvec, WPvec, label = r'Sticky Wage 1')

    plt.title('Labour Market')
    plt.xlabel(r'Labour ($N$)')
    plt.ylabel(r'Real Wage ($\frac{W}{P}$)')

    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.legend(loc = 'upper right')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.clf()

    

# Summary:{{{1
def modelfiguretext(modelname, graph1, graph2, graph3, graph4, graph5):
    """
    Return model figure tex with correct locations.
    """
    tex = modelfiguretex
    tex = tex.replace('graph1.jpg', graph1)
    tex = tex.replace('graph2.jpg', graph2)
    tex = tex.replace('graph3.jpg', graph3)
    tex = tex.replace('graph4.jpg', graph4)
    tex = tex.replace('graph5.jpg', graph5)

    tex = tex.replace('modelname', modelname)

    return(tex)


def changesastext(before, after, variables):
    text = 'We observe the following changes in the endogenous variables at equilibrium: '
    for variable in variables:
        change = after[variable] - before[variable]
        if change > 0.0001:
            changemessage = "increases"
        elif change < -0.0001:
            changemessage = "decreases"
        else:
            changemessage = "stays the same"
        # text = text + '$' + variable + '_0: ' + str(before[variable]) + ', ' + variable + '_1: ' + str(after[variable]) + '$. '
        text = text + '$' + variable + '$ ' + changemessage + '. '

    return(text)


def classical_full(before, after = None, savedir = None, modelnamesuffix = None):
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    before = classicalequilibrium(before)
    after = classicalequilibrium(after)

    labourclassical_graph(before, after = after, savedir = savedir)
    asad_graph(before, after = after, savedir = savedir)
    output_graph(before, after = after, savedir = savedir)
    islm_graph(before, after = after, savedir = savedir)
    money_graph(before, after = after, savedir = savedir)

    if savedir is not None and after is not None:
        text = changesastext(before, after, ['i', 'N', 'P', 'W', 'Y'])

        if modelnamesuffix is None:
            modelname = 'Classical Model'
        else:
            modelname = 'Classical Model following ' + modelnamesuffix

        modelfigure = modelfiguretext(modelname, savedir + 'islm.jpg', savedir + 'money.jpg', savedir + 'asad.jpg', savedir + 'labourclassical.jpg', savedir + 'output.jpg')

        with open(savedir + 'summary.tex', 'w+') as f:
            f.write(text + '\n\n' + modelfigure)

        with open(savedir + 'figures.tex', 'w+') as f:
            f.write(modelfigure)


def stickynominalwage_full(before, after = None, savedir = None, modelnamesuffix = None):
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    before = stickynominalwageequilibrium(before)
    after = stickynominalwageequilibrium(after)

    laboursticky_graph(before, after = after, savedir = savedir)
    asad_graph(before, after = after, savedir = savedir)
    output_graph(before, after = after, savedir = savedir)
    islm_graph(before, after = after, savedir = savedir)
    money_graph(before, after = after, savedir = savedir)

    if savedir is not None and after is not None:
        text = changesastext(before, after, ['i', 'N', 'P', 'Y'])

        if modelnamesuffix is None:
            modelname = 'Sticky Nominal Wage Model'
        else:
            modelname = 'Sticky Nominal Wage Model following ' + modelnamesuffix

        modelfigure = modelfiguretext(modelname, savedir + 'islm.jpg', savedir + 'money.jpg', savedir + 'asad.jpg', savedir + 'laboursticky.jpg', savedir + 'output.jpg')

        with open(savedir + 'summary.tex', 'w+') as f:
            f.write(text + '\n\n' + modelfigure)

        with open(savedir + 'figures.tex', 'w+') as f:
            f.write(modelfigure)


def stickyrealwage_full(before, after = None, savedir = None, modelnamesuffix = None):
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    before = stickyrealwageequilibrium(before)
    after = stickyrealwageequilibrium(after)

    laboursticky_graph(before, after = after, savedir = savedir)
    asad_graph(before, after = after, savedir = savedir)
    output_graph(before, after = after, savedir = savedir)
    islm_graph(before, after = after, savedir = savedir)
    money_graph(before, after = after, savedir = savedir)

    if savedir is not None and after is not None:
        text = changesastext(before, after, ['i', 'N', 'P', 'W', 'Y'])

        if modelnamesuffix is None:
            modelname = 'Sticky Real Wage Model'
        else:
            modelname = 'Sticky Real Wage Model following ' + modelnamesuffix

        modelfigure = modelfiguretext(modelname, savedir + 'islm.jpg', savedir + 'money.jpg', savedir + 'asad.jpg', savedir + 'laboursticky.jpg', savedir + 'output.jpg')

        with open(savedir + 'summary.tex', 'w+') as f:
            f.write(text + '\n\n' + modelfigure)

        with open(savedir + 'figures.tex', 'w+') as f:
            f.write(modelfigure)


def allmodels(before, after = None, savedir = None, modelnamesuffix = None):

    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    classical_full(before, after = after, savedir = savedir + 'classical/', modelnamesuffix = modelnamesuffix)
    stickynominalwage_full(before, after = after, savedir = savedir + 'stickynominalwage/', modelnamesuffix = modelnamesuffix)
    stickyrealwage_full(before, after = after, savedir = savedir + 'stickyrealwage/', modelnamesuffix = modelnamesuffix)
    
