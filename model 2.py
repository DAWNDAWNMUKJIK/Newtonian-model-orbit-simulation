import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy import units as un
from astropy.constants import G, M_sun, R_sun

#set up the intial value of mass and radius of binary stars. 
# Setting up the intial seperation between two star as standard distance between sun and earth for convention.

m_1 = 1.0 * M_sun

m_2 = 1.0 * M_sun

R_1 = 1.0 * R_sun

R_2 = 1.0 * R_sun

R_sep = 1.0 * un.AU

#unit conversion of Standard Si unit of G to AU, Solar mass and radius

G_au = G.to(un.AU**3 / (un.M_sun * un.year**2))

m_1_solar = m_1.to(un.M_sun)

m_2_solar = m_2.to(un.M_sun)

mu = G_au * (m_1_solar + m_2_solar)

mu_num = mu.to_value(un.AU**3 / un.year**2)

#velocity compoentn with respect to radius of seperation

v_circ = np.sqrt(mu / R_sep).to(un.AU / un.year)
v_esc = np.sqrt(2 * mu / R_sep).to(un.AU / un.year)

#setting up the newtonian model for orbital motions
def newtonian_model(t,Y):
    x, y, v_x, v_y = Y
    dx = v_x
    dy = v_y
    r = np.sqrt(x ** 2 + y ** 2)
    dv_x = - mu_num * x/r**3
    dv_y = - mu_num * y/r**3
    return [dx, dy, dv_x, dv_y]

#For which case it is possible to have collsion orbit

coll_d = (R_1 + R_2).to_value(un.AU)

def coll (t,Y):
    x, y, v_x, v_y = Y
    r = np.sqrt(x ** 2 + y ** 2)

    return r - coll_d

coll.terminal = True
coll.direction = -1

# case factor for each possible outcome
case = { 'collision orbit': (0, 0), 'Bound orbit': (0, 0.7 * v_circ.value), "circular orbit": (0, v_circ.value), "escape orbit": (0, 1.1 * v_esc.value)}

# mass fraction of star's positions

f_m1 = (m_2_solar / (m_1_solar + m_2_solar)).value

f_m2 = (m_1_solar / (m_1_solar + m_2_solar)).value

#plot the simulated orbit of each possible cases
t_span = [0, 3]
plt.figure(figsize=(5, 5))


for name, (v_x0, v_y0) in case.items():

    Y0 = [R_sep.to_value(un.AU), 0, v_x0, v_y0]

    soln = solve_ivp(newtonian_model, t_span, Y0, t_eval = np.linspace(0, 3, 3000), events = coll, rtol = 1e-9, atol =  1e-11)

# Relative position
    x = soln.y[0]
    y = soln.y[1]

# relative position from model 2 to actual position of each star

    x1 = f_m1 * x
    y1 = f_m1 * y

    x2 = -f_m2 * x
    y2 = -f_m2 * y

    plt.plot(x1, y1, label=f"{name} - Star 1")

    plt.plot(x2, y2, linestyle="--", label=f"{name} - Star 2")


plt.scatter([0], [0], marker="o", s = 100, label="CoM")
plt.xlabel("x [AU]")
plt.ylabel("y [Au]")
plt.title("Binary star's Esacape, Bound, and collision orbit")
plt.axis("equal")
plt.xlim( -0.7, 0.7)
plt.ylim( -0.5, 0.5)
plt.legend(loc = 'upper right')
plt.grid(ls = '-')
plt.show()




    