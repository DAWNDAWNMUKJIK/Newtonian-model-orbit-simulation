import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from astropy import constants as ap

#setting mu = 1 for simple binary orbit to produce approximated circular orbit
#For simplicity we set m1,m2 = 1solar mass
#consider the two circular orbit problem since we are dealing with binary system with mass both as 1 solar mass for simplicity
m_1 = 1.0
m_2 = 2.0
G = 1.0
mu = G * (m_1 + m_2)
#setting up the newtonian model for simple circular orbit=
def newtonian_model(t,Y):
    x, y, v_x, v_y = Y
    dx = v_x
    dy = v_y
    r = np.sqrt(x**2 + y**2)
    dv_x = -mu*(x/r**(3/2))
    dv_y =  -mu*(y/r**(3/2))
    return [dx, dy, dv_x, dv_y]


#set newtonian model's value into [1,0,0,v_y, where v_y is [0.0, 0.5, 1.0, 1.5] to test all possible case and set up the span and its linearspace
t_span = [0, 20]
t_eval = np.linspace(0, 20, 3000)
v_y = [0.0, 0.5, 1.0, 1.5, 2.0]

for v_y_0 in v_y:
    Y_0 = [1,0,0,v_y_0]
    soln = solve_ivp(newtonian_model, t_span, Y_0, t_eval = t_eval, rel_error = 1e-10, abs_error = 1e-12)
    x = soln.y[0]
    y = soln.y[1]
    
    plt.plot(x, y, label = "star_velocity = " + str(v_y_0) + "v_circ" )


plt.scatter([0], [0], marker="o", s = 100, label="Center of Mass")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend(loc = 'upper right')
plt.grid(ls = '-')
plt.show()




    