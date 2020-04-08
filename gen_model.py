import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import random

N = 329500000
D = [1, 1, 2, 2, 5, 5, 5, 5, 5, 7, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 51, 51, 57, 58, 60, 68, 74, 98, 118, 149, 217, 262, 402, 518, 583, 959, 1281, 1663, 2179, 2727, 3499, 4632, 6421, 7783, 13677, 19100, 25489, 33276, 43847, 53740, 65778, 83836, 101657, 121478, 140886, 161807, 188172, 213372, 243453, 275586, 308850, 337072]

# N = 763
# D = [1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4]

def fit_sir(data, population, end_period, intv=50):
    def error(params):
        b, g = params
        
        test = solve_ivp(
            lambda t, m: [(-b*m[0]*m[1]/population), (b*m[0]*m[1]/population) - (g*m[1]), (g*m[1])],
            [0, len(data)], [population - data[0], data[0], 0], 
            t_eval=np.arange(0, len(data), 1/intv)
        )

        return sum((test.y[1][::intv] - data)**2)

    while True:
        opt_params = minimize(error, [1 + random.random(), 1 + random.random()], method="Nelder-Mead")
        beta, gamma = opt_params.x

        if beta < 0 or gamma < 0:
            continue

        model = solve_ivp(
            lambda t, m: [(-beta*m[0]*m[1]/population), (beta*m[0]*m[1]/population) - (gamma*m[1]), (gamma*m[1])], 
            [0, len(data) + end_period], 
            [population - data[0], data[0], 0], 
            t_eval=np.arange(0, len(data) + end_period, 1/intv)
        )

        rsq = 1 - (np.var(model.y[1][::intv][:-end_period] - data) / np.var(data))

        if rsq < 0.1:
            continue
        else:
            break

    return model, beta, gamma, rsq

"""
model, beta, gamma, rsq = fit_sir(D, N, 30)
plt.plot(model.t, model.y[1], "r-")
plt.plot(np.arange(len(D)), D, "*:")
plt.show()
code.interact(local=locals())
"""