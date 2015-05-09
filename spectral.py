"""Spectral method for evolving GPE for BEC's"""

from numpy import exp, pi, arange, meshgrid, sqrt, linspace
from numpy.fft import fft2, ifft2, fftshift
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

matplotlib.rcParams['figure.dpi'] = 200

from timestep2d import QHO


class Simulation:
    """Simulation to step wavefunction forward in time from the given parameters
xmax : maximum extent of boundary
N    : number of spatial points
init : initial wavefunction
nonlinearity : factor in front of |psi^2| term
"""
    def __init__(self, parameters):
        self.parameters = parameters

        # set up spatial dimensions
        xmax = parameters['xmax']
        self.xmax = xmax
        N = parameters['N']
        v = linspace(-xmax, xmax, N)
        self.dx = v[1] - v[0]
        self.x, self.y = meshgrid(v, v)

        # spectral space
        kmax = 2*pi / self.dx
        dk = kmax / N
        self.k = fftshift((arange(N)-N/2) * dk)
        kx, ky = meshgrid(self.k, self.k)

        # time
        self.steps = 0
        self.time = 0
        self.dt = self.dx**2 / 4

        # wavefunction
        init_func = parameters['initial']
        self.wf = init_func(self.x, self.y, 0)
        self.wf /= sqrt(self.norm().sum() * self.dx**2) # normalize

        # Hamiltonian operators
        self.loss = 1 - 1j*parameters['loss']
        self.T = exp(-1j * self.loss * (kx**2 + ky**2) * self.dt / 2)
        self.V = exp(-1j * self.loss * (self.x**2 + self.y**2) * self.dt / 2)
        self.eta = parameters['nonlinearity']


    def evolve(self, time):
        """Evolve the wavefunction to the given time in the future"""
        steps = int(time / self.dt)
        if steps == 0:
            steps = 1 # guarantee at least 1 step

        for _ in range(steps):
            #self.linear_step()
            self.nonlinear_step()
            if self.loss:
                N = self.norm().sum()*self.dx**2
                self.wf /= N

        self.update_time(steps)

        
    def linear_step(self):
        """Make one linear step dt forward in time"""
        # kinetic
        self.wf[:] = fft2(ifft2(self.wf) * self.T)

        # potential
        self.wf *= self.V


    def nonlinear_step(self):
        """Make one nonlinear step dt forward in time"""
        # linear step
        self.linear_step()

        # nonlinear
        self.wf *= exp(-1j * self.loss * self.eta * abs(self.wf)**2 * self.dt)


    def update_time(self, steps):
        """Increment time by steps taken"""
        self.steps += steps
        self.time = self.steps * self.dt

    def norm(self):
        return abs(self.wf)**2


    def show(self):
        """Show the current norm of the wavefunction"""
        fig, ax = plt.subplots()
        ax.imshow(self.norm(), cmap=plt.cm.hot)
        plt.show()



def animate(simulation, time, interval=100):
    """Display an animation of the simulation"""
    fig, ax = plt.subplots()
    L = simulation.xmax
    norm = ax.imshow(simulation.norm(), extent=(-L, L, -L, L), cmap=plt.cm.hot)

    def update(i):
        simulation.evolve(time / interval)
        N = simulation.norm()
        norm.set_data(N)
        ax.set_title('T = {:3.2f}, N = {:1.6f}'.format(simulation.time, N.sum()*simulation.dx**2))

    anim = animation.FuncAnimation(fig, update, interval=10)
    plt.show()


    

if __name__ == '__main__':
    params = {'N': 800,
              'xmax': 7,
              'nonlinearity': 4,
              'initial': QHO(n=0, xshift=0), # GaussianWavepacket(1, 5, -4),
              'loss': 0.0,
              }

    sim = Simulation(params)
    #animate(sim, 1)
    #sim.evolve()
    #sim.wf.dump('wf')

