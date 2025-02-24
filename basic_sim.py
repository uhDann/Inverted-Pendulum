import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sympy import *

# create the symbolic parameter variables
m1,m2,l,g=symbols('m1 m2 l g')
# symbolic force
F=symbols('F')
# state variables
z1,z2,z3,z4=symbols('z1 z2 z3 z4')

# derivatives
dz2, dz4 =symbols('dz2 dz4')

# first equation
e1=(m1+m2)*dz2-m2*l*dz4*cos(z3)+m2*l*(z4**2)*sin(z3)-F
# second equation
e2=l*dz4-dz2*cos(z3)-g*sin(z3)

# define and solve the equations
result=solve([e1, e2], dz2,dz4 , dict=True)

# first equation
dz2Solved=simplify(result[0][dz2])

# second equation
dz4Solved=simplify(result[0][dz4])

# define the numerical values of the parameters
lV=1
gV=9.81

m1V=10
m2V=1
# substituted them in the equations
dz2Solved=dz2Solved.subs(l,lV).subs(g,gV).subs(m1,m1V).subs(m2,m2V)
dz4Solved=dz4Solved.subs(l,lV).subs(g,gV).subs(m1,m1V).subs(m2,m2V)

# create the Python functions that return the numberical values 
functionDz2=lambdify([z1,z2,z3,z4,F],dz2Solved)
functionDz4=lambdify([z1,z2,z3,z4,F],dz4Solved)

# this function defines the state-space model, that is, its right-hand side
# z is the state 
# t is the current time - internal to solver
# timePoints - time points vector necessary for interpolation
# forceArray - time-varying input
def stateSpaceModel(z,t,timePoints,forceArray):
    # interpolate input force values
    # depending on the current time
    forceApplied=np.interp(t,timePoints, forceArray)
    dz2Value=functionDz2(z[0],z[1],z[2],z[3],forceApplied)
    dz4Value=functionDz4(z[0],z[1],z[2],z[3],forceApplied)
    dzdt=[z[1],dz2Value,z[3],dz4Value]
    return dzdt


# define the simulation parameters
startTime=0
endTime=25
timeSteps=5000
 
# simulation time array 
# we will obtain the solution at the time points defined by 
# the vector simulationTime
simulationTime=np.linspace(startTime,endTime,timeSteps)
 
# define the force input 
#forceInput = np.sin(simulationTime)+np.cos(2*simulationTime)
forceInput = np.zeros(shape=(simulationTime.shape))

# plot the applied force
plt.plot(simulationTime, forceInput)
plt.xlabel('time')
plt.ylabel('Force - [N]')
plt.savefig('inputSequence.png',dpi=600)
plt.show()

# define the initial state for simulation 
initialState=np.array([0,0,np.pi/3,0])

# generate the state-space trajectory by simulating the state-space model
solutionState=odeint(stateSpaceModel,initialState,
                     simulationTime,
                     args=(simulationTime,forceInput))

# save the simulation data 
# the save file is opened by another Python script that is used
# to animate the trajectory
np.save('simulationData.npy', solutionState)

# plot the state trajectories 
# cart state trajectories
plt.figure(figsize=(10,8))
plt.plot(simulationTime, solutionState[:,0],'b',linewidth=4,label='z1')
plt.plot(simulationTime, solutionState[:,1],'r',linewidth=4,label='z2')
plt.xlabel('time',fontsize=16)
plt.ylabel('Cart states',fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid()
plt.savefig('cartStates.png',dpi=600)
plt.show()

# plot the state trajectories 
# pendulum state trajectories
plt.figure(figsize=(10,8))
plt.plot(simulationTime, solutionState[:,2],'b',linewidth=4,label='z3')
plt.plot(simulationTime, solutionState[:,3],'r',linewidth=4,label='z4')
plt.xlabel('time',fontsize=16)
plt.ylabel('Pendulum states',fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid()
plt.savefig('pendulumStates.png',dpi=600)
plt.show()
