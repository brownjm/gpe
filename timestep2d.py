"""Modified Visscher's method for Gross-Pitaevskii equation for BEC's

Original paper:
A fast explicit algorithm for the time-dependent Schroedinger equation
P. B. Visscher
Computers in Physics 5, 596 (1991)
doi: 10.1063/1.168415
"""

from math import factorial
from numpy import pi, sqrt, exp, linspace, meshgrid, zeros, empty, random, cosh, arctan2
import numpy

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
        self.xmax = xmax
        N = parameters['N']
        self.v = linspace(-xmax, xmax, N)
        self.dx = self.v[1] - self.v[0]
        print("dx = {}".format(self.dx))
        self.x, self.y = meshgrid(self.v, -self.v)

        # time
        self.steps = 0
        self.time = 0
        self.dt = self.dx**2 / 4 # good choice for stability
        print("dt = {}".format(self.dt))

        # wavefunction
        init_func = parameters['initial']
        self.wf = Wavefunction(init_func, self.x, self.y, self.dt)
        
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
        self.wf = Wavefunction(init_func, self.x, self.y, self.dt)
        
        
    def add_observer(self, observer):
        """Add an observer that will be given the wavefunction after each call of evolve"""
        self.observers.append(observer)

    
    def evolve(self, time):
        """Evolve the wavefunction to the given time in the future"""
        steps = int(time / self.dt)
        #print("n = {}".format(steps))
        if steps == 0: 
            steps = 1 # guarantee at least 1 step
        for _ in range(steps):
            #self.linear_step()
            self.nonlinear_step()
            
        self.update_time(steps)

        for ob in self.observers:
            ob.notify(self.time, self.wf)


    def linear_step(self):
        """Make one linear step dt forward in time"""
        # shorten the names of variables
        x = self.x
        y = self.y
        t = self.time
        dt = self.dt
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V


        # update wavefunctions
        real += dt * (T.fast(imag) + V(x, y, t) * imag)
        prev[:] = imag[:] # need to explicitly copy values, assignment doesn't work
        
        imag -= dt * (T.fast(real) + V(x, y, t+dt/2) * real)


    def nonlinear_step(self):
        """Make one nonlinear step dt forward in time"""
        # shorten the names of variables
        x = self.x
        y = self.y
        t = self.time
        dt = self.dt
        eta = self.eta
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V


        # update wavefunctions
        real[:] = (real + dt*(T.fast(imag) + V(x, y, t)*imag + eta*imag*imag*imag)) / (1 - dt*eta*real*imag)
        prev[:] = imag # need to explicitly copy values, assignment doesn't work
        
        imag[:] = (imag - dt*(T.fast(real) + V(x, y, t+dt/2)*real + eta*real*real*real)) / (1 + dt*eta*real*imag)


    def update_time(self, steps):
        """Increment time by steps taken"""
        self.steps += steps
        self.time = self.steps * self.dt

    def show(self):
        L = self.xmax
        plt.imshow(self.wf.norm(), plt.cm.hot, extent=(-L, L, -L, L))
        plt.colorbar()
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
        new[1:-1, 1:-1] = self.coef * (wf[:-2, 1:-1] + wf[2:, 1:-1]
                                       + wf[1:-1, :-2] + wf[1:-1, 2:]
                                       - 4*wf[1:-1, 1:-1])
        new[:,0] = new[:,-1] = new[0,:] = new[-1,:] = 0.0
        return new


class Potential(object):
    """Potential V in Hamiltonian"""
    def __init__(self, potential_name):
        self.func = self.__getattribute__(potential_name)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def free(self, x, y, t):
        return zeros(x.shape)
    
    def linear(self, x, y, t):
        return -x

    def harmonic(self, x, y, t):
        return 0.5*(x*x + y*y)

    def barrier(self, x, y, t):
        potential = zeros(x.shape, dtype=float)
        inside = abs(x) < 0.1
        potential[inside] = 10
        potential[~inside] = 0
        return potential

    def laser_swipe(self, x, y, t):
        harmonic = 0.5 * (x*x + y*y)
        w = 0.1
        v = 20
        xs = x - v*t + 5
        laser = 100*exp(-(xs*xs + y*y) / w**2)
        return harmonic + laser
        

class Wavefunction:
    """Complex valued wavefunction where real and imaginary values live at different times"""
    def __init__(self, init_func, x, y, dt):
        # real R(x,0) and imaginary I(x,dt/2) part of wavefunction
        self.real = init_func(x, y, 0).real
        self.imag = init_func(x, y, dt/2).imag

        # stores previous imaginary part
        self.prev = init_func(x, y, -dt/2).imag

        # normalize wavefunction
        dx = x[0,1] - x[0,0]
        N = sqrt(abs(self.norm()).sum() * dx**2)
        self.real /= N
        self.imag /= N
        self.prev /= N

    def norm(self):
        """|psi|^2 at integer t/dt times"""
        return self.real**2 + self.imag*self.prev


class GaussianWavepacket:
    """Gaussian wavepacket"""
    def __init__(self, width, k, xshift=0, yshift=0):
        self.w = width
        self.k = k
        self.xshift = xshift
        self.yshift = yshift

    def __call__(self, x, y, t):
        xs = x - self.xshift
        ys = y - self.yshift
        k = self.k
        w = self.w
        wf = exp(-(xs**2 + ys**2) / w**2 + 1j*(k*x - k**2 * t / 2))
        return wf

class Sech:
    """Sech solution for harmonic trap"""
    def __init__(self, mu=-0.5):
        self.mu = mu

    def __call__(self, x, y, t):
        return 1/sqrt(2) * exp(-1j*self.mu*t) / cosh(x*x + y*y)


class QHO:
    """Quantum harmonic oscillator wavefunctions"""
    def __init__(self, n, xshift=0, yshift=0):
        self.n = n
        self.xshift = xshift
        self.yshift = yshift
        self.E = n + 0.5
        self.coef = 1 / sqrt(2**n * factorial(n)) * (1 / pi)**(1/4)
        self.hermite = hermite(n)

    def __call__(self, x, y, t):
        xs = x - self.xshift
        ys = y - self.yshift
        return self.coef * exp(-(xs**2 + ys**2) / 2 - 1j*self.E*t) * self.hermite(x) * self.hermite(y)

class Noise:
    """Random floating point noise that falls between 0 and 1"""
    def __call__(self, x, y, t):
        return random.ranf(x.shape) + 1j*random.ranf(x.shape)

class File:
    """Load a wavefunction from file"""
    def __init__(self, filename):
        self.filename = filename
        self.wf = numpy.load(filename)

    def __call__(self, x, y, t):
        return self.wf

class Vortex:
    """Load a wavefunction from file and add vortex"""
    def __init__(self, filename):
        self.filename = filename
        self.wf = numpy.load(filename)
        
    def __call__(self, x, y, t):
        a = 20
        shift = 0.3
        xs = x - shift
        rho = sqrt(xs*xs + y*y)
        f = (rho / a) / sqrt(1 + (rho/a)**2)
        theta = arctan2(y, xs)

        return self.wf * f * exp(1j*theta)

        
class Observer:
    def notify(self, time, wf):
        raise NotImplementedError("Overload this method in the derived observer")

class WavefunctionObserver(Observer):
    def __init__(self):
        self.time = []
        self.wavefunction = []
        
    def notify(self, time, wf):
        self.time.append(time)
        self.wavefunction.append(wf)
        
        
def animate(simulation, time, interval=100):
    """Display an animation of the simulation"""
    wf = simulation.wf
    
    
    fig, ax = plt.subplots()
    L = simulation.xmax
    norm = ax.imshow(wf.norm(), extent=(-L, L, -L, L), cmap=plt.cm.hot)
    #V = ax.imshow(simulation.V(simulation.x, simulation.y, simulation.time))

    def update(i):
        simulation.evolve(time / interval)
        N = wf.norm()
        norm.set_data(N)
        ax.set_title('T = {:3.2f}, N = {:1.6f}'.format(simulation.time, N.sum() * simulation.dx**2))
        # V.set_data(simulation.V(simulation.x, simulation.y, simulation.time))
        # ax.set_title('T = {:3.2f}'.format(simulation.time))

        
    anim = animation.FuncAnimation(fig, update, interval=10)
    plt.show()


def make_movie(simulation, time):
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='GPE 2D', artist='matplotlib',
                    comment="Adapted Visscher's taggered time step method")
    writer = FFMpegWriter(fps=15, metadata=metadata)


    
    fig, ax = plt.subplots()
    wf = simulation.wf
    L = sim.xmax
    norm = ax.imshow(wf.norm(), extent=(-L, L, -L, L), cmap=plt.cm.hot)

    steps = int(time/sim.dt)
    with writer.saving(fig, "test.mp4", 100):
        for _ in range(int(steps/100)):
            simulation.evolve(steps*sim.dt/100)
            N = wf.norm()
            norm.set_data(N)
            writer.grab_frame()


if __name__ == '__main__':
    params = {'N': 128,
              'xmax': 7,
              'BC': 'reflect',
              'nonlinearity': 4,
              'initial': Vortex('wf.dat'),
              'potential': 'harmonic',
              }

    sim = Simulation(params)
    #wfob = WavefunctionObserver()
    #sim.add_observer(wfob)
    #sim.evolve(0.1)
    #sim.show()
    animate(sim, 5)
    #make_movie(sim, 40)
