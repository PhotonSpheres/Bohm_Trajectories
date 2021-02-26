from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

## Set up the ODE from Bohmian Mechanics with given psi as a sum of two Gaussian 
## wave packets (hence a solution of the Schrödinger eq.) centered around -1; +1

def Bohm_Trajec( Q, t, h, m, sigma ):
    lambda_t = 1 + ( 1j * h * t ) / ( 2 * m * sigma * sigma )
    factor = ( 2 * np.pi * lambda_t * lambda_t * sigma * sigma )**( - 3 / 4 )
    gauss_1 = factor * np.exp( - ( Q - 1 )**2 / ( 4 * lambda_t * sigma * sigma ) )
    gauss_2 = factor * np.exp( - ( Q + 1 )**2 / ( 4 * lambda_t * sigma * sigma ) )
    dx_gauss1 = - ( ( 2 * Q - 2 ) / ( 4 * lambda_t * sigma * sigma ) ) * gauss_1
    dx_gauss2 = - ( ( 2 * Q + 2 ) / ( 4 * lambda_t * sigma * sigma ) ) * gauss_2
    psi = ( 1 / np.sqrt(2) ) * ( gauss_1 + gauss_2 )
    dx_psi = ( 1 / np.sqrt(2) ) * ( dx_gauss1 + dx_gauss2 )

    dt_Q = ( h / m ) * np.imag( dx_psi / psi )

    return dt_Q


## Specify the time grid and the parameters of the Gaussian wave packets

time = np.linspace(0,0.25,300) # time grid
h = 1 # Planck constant
m = 1 # mass
sigma = 0.05 


## Solve the ODE using odeint from scipy. Intial Data revolves +0.11; -0.11 around the centers
## +1; -1 of the Gaussian functions

for Q0 in np.linspace( -0.89, -1.11, 40 ):
    sol = odeint( Bohm_Trajec, Q0, time, args=(h, m, sigma) )
    plt.plot(time, sol)

for Q0 in np.linspace( 0.89, 1.11, 40 ):
    sol = odeint( Bohm_Trajec, Q0, time, args=(h, m, sigma) )
    plt.plot(time, sol)


# Shot the result

plt.ylim(-1.6, 1.6)
plt.show()