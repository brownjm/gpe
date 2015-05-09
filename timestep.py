"""Modified Visscher's method for Gross-Pitaevskii equation for BEC's

Original paper:
A fast explicit algorithm for the time-dependent Schroedinger equation
P. B. Visscher
Computers in Physics 5, 596 (1991)
doi: 10.1063/1.168415
"""

from math import factorial
from numpy import pi, sqrt, exp, linspace, zeros, empty, cosh

from scipy.ndimage.filters import laplace
from scipy.special import hermite

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


# higher resolution plots
matplotlib.rcParams['figure.dpi'] = 200


class Simulation:
    """Simulator steps a wavefunction forward in time from the given parameters
xmax : maximum extent of boundary
N    : number of spatial points
BC   : boundary conditions, e.g. reflect, wrap
init : initial wavefunction
potential    : potential name V(x), e.g. linear, harmonic
nonlinearity : factor in front of |psi^2| term
"""
    def __init__(self, parameters):
        self.parameters = parameters
        
        # set up spatial dimensions
        xmax = parameters['xmax']
        N = parameters['N']
        self.x = linspace(-xmax, xmax, N)
        self.dx = self.x[1] - self.x[0]

        # time
        self.steps = 0
        self.time = 0
        self.dt = self.dx**2 / 4 # good choice for stability
        
        # wavefunction
        init_func = parameters['initial']
        self.wf = Wavefunction(init_func, self.x, self.dt)
        
        # Hamiltonian operators
        BC = parameters['BC']
        self.T = Kinetic(self.dx, BC)
        self.V = Potential(parameters['potential'])

        # nonlinearity
        self.eta = parameters['nonlinearity']

        # observers
        self.observers = []


    def reset(self):
        """Reinitializes wavefunction and sets time = 0"""
        self.steps = 0
        self.time = 0
        init_func = self.parameters['initial']
        self.wf = Wavefunction(init_func, self.x, self.dt)
        
        
    def add_observer(self, observer):
        """Add an observer that will be given the wavefunction after each call of evolve"""
        self.observers.append(observer)

    
    def evolve(self, time):
        """Evolve the wavefunction to the given time in the future"""
        steps = int(time / self.dt)
        if steps == 0:
            steps = 1 # guarantee at least 1 step
        for _ in range(steps):
            #self.linear_step()
            self.nonlinear_step()
        
        self.update_time(steps)

        for ob in self.observers:
            ob.notify(self.wf)


    def linear_step(self):
        """Make one linear step dt forward in time"""
        # shorten the names of variables
        x = self.x
        dt = self.dt
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V

        # update wavefunctions
        real += dt * (T.fast(imag) + V(x) * imag)
        prev[:] = imag # need to explicitly copy values, assignment doesn't work
        imag -= dt * (T.fast(real) + V(x) * real)

        
    def nonlinear_step(self):
        """Make one nonlinear step dt forward in time"""
        # shorten the names of variables
        x = self.x
        dt = self.dt
        eta = self.eta
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V

        # update wavefunctions
        real[:] = (real + dt*(T.fast(imag) + V(x)*imag + eta*imag*imag*imag)) / (1 - dt*eta*real*imag)
        prev[:] = imag # need to explicitly copy values, assignment doesn't work
        imag[:] = (imag - dt*(T.fast(real) + V(x)*real + eta*real*real*real)) / (1 + dt*eta*real*imag)



    def update_time(self, steps):
        """Increment time by one step in dt"""
        self.steps += steps
        self.time = self.steps * self.dt

    def plot(self):
        plt.plot(self.wf.real, label='real')
        plt.plot(self.wf.real, label='imag')
        plt.plot(self.wf.norm(), label='norm')
        plt.legend()
        plt.show()



class Kinetic:
    """Kinetic energy term in Hamiltonian"""
    def __init__(self, dx, BC='reflect'):
        self.coef = -0.5 / dx**2
        self.BC = BC

    def __call__(self, wf):
        return self.coef * laplace(wf, mode=self.BC)

    def fast(self, wf):
        new = empty(wf.shape)
        new[1:-1] = self.coef * (wf[:-2] + wf[2:] - 2*wf[1:-1])
        new[0] = new[-1] = 0.0
        return new


class Potential:
    """Potential V in Hamiltonian"""
    def __init__(self, potential_name):
        self.func = self.__getattribute__(potential_name)
        self.past = None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


    def free(self, x):
        return zeros(x.shape)
    
    def linear(self, x):
        return -x

    def harmonic(self, x):
        return 0.5*x**2

    def barrier(self, x):
        potential = zeros(x.shape, dtype=float)
        inside = abs(x) < 3
        potential[inside] = -100
        potential[~inside] = 0
        return potential
        

class Wavefunction:
    """Complex valued wavefunction where real and imaginary values live at different times"""
    def __init__(self, init_func, x, dt):
        # real R(x,0) and imaginary I(x,dt/2) part of wavefunction
        self.real = init_func(x, 0).real
        self.imag = init_func(x, dt/2).imag

        # stores previous imaginary part
        self.prev = init_func(x, -dt/2).imag

        # normalize wavefunction
        dx = x[1] - x[0]
        N = sqrt(sum(abs(self.norm()))*dx)
        self.real /= N
        self.imag /= N
        self.prev /= N

    def norm(self):
        """|psi|^2 at integer t/dt times"""
        return self.real**2 + self.imag*self.prev

class Sech:
    """Sech(x) solution"""
    def __init__(self, mu=-0.5):
        self.mu = mu

    def __call__(self, x, t):
        return 1/sqrt(2) * exp(-1j*self.mu*t) / cosh(x)

class GaussianWavepacket:
    """Gaussian wavepacket"""
    def __init__(self, width, k, shift=0):
        self.w = width
        self.k = k
        self.shift = shift

    def __call__(self, x, t):
        xs = x - self.shift
        k = self.k
        w = self.w
        wf = exp(-xs**2 / w**2 + 1j*(k*x - k**2 * t / 2))
        return wf


class QHO:
    """Quantum harmonic oscillator wavefunctions"""
    def __init__(self, n, shift=0):
        self.n = n
        self.shift = shift
        self.E = n + 0.5
        self.coef = 1 / sqrt(2**n * factorial(n)) * (1 / pi)**(1/4)
        self.hermite = hermite(n)

    def __call__(self, x, t):
        xs = x - self.shift
        return self.coef * exp(-xs**2 / 2 - 1j*self.E*t) * self.hermite(x)
        

        
def animate(simulation, time, interval=100):
    """Display an animation of the simulation"""
    x = simulation.x
    V = simulation.V
    wf = simulation.wf
    
    fig, ax = plt.subplots()

    # plot scaled potential
    potential = V(x)
    if max(abs(potential)) != 0.0:
        potential /= max(abs(potential))
    ax.plot(x, potential, 'k')

    # real, = ax.plot(x, wf.real, label='real')
    # imag, = ax.plot(x, wf.imag, label='imag')
    norm, = ax.plot(x, wf.norm(), label='norm')

    def update(i):
        simulation.evolve(time / interval)
        # real.set_ydata(wf.real)
        # imag.set_ydata(wf.imag)
        N = wf.norm()
        norm.set_ydata(N)
        ax.set_title('T = {:3.2f}, N = {:1.6f}'.format(sim.time, sum(N)*sim.dx))

        
    anim = animation.FuncAnimation(fig, update, interval=10)
    plt.show()



if __name__ == '__main__':
    params = {'N': 2001,
              'xmax': 25,
              'BC': 'reflect',
              'nonlinearity': 0,
              'initial': GaussianWavepacket(1, 5, -4),#Sech(), #QHO(0), # 
              'potential': 'barrier',
              }

    sim = Simulation(params)
    animate(sim, 1)
