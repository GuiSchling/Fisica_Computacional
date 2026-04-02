import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define a função que representa o sistema de EDOs de primeira ordem
# y'' + b*y' + y = 0
# Seja Y = [y, v], então Y' = [v, -b*v - y]
def model(Y, t, b):
    y, v = Y
    dydt = [v, -b*v - y]
    return dydt

# Condições iniciais: solto de y0 = 1 com velocidade inicial 0
Y0 = [1, 0]

# Tempo para a simulação
t = np.linspace(0, 15, 500)

# Resolver para diferentes valores de b
# Subamortecido (b < 2)
b_sub = 0.5
sol_sub = odeint(model, Y0, t, args=(b_sub,))

# Criticamente amortecido (b = 2)
b_crit = 2
sol_crit = odeint(model, Y0, t, args=(b_crit,))

# Superamortecido (b > 2)
b_super = 5
sol_super = odeint(model, Y0, t, args=(b_super,))

# Plotagem
plt.figure(figsize=(10, 6))
plt.plot(t, sol_sub[:, 0], 'b', label=f'Subamortecido (b={b_sub}, $\Delta < 0$)')
plt.plot(t, sol_crit[:, 0], 'g', label=f'Crítico (b={b_crit}, $\Delta = 0$)', linewidth=2)
plt.plot(t, sol_super[:, 0], 'r', label=f'Superamortecido (b={b_super}, $\Delta > 0$)')

# Linha de equilíbrio
plt.axhline(0, color='black', linestyle='--', alpha=0.3)

plt.title('Tipos de Amortecimento no Sistema Massa-Mola ($y\'\' + b y\' + y = 0$)')
plt.xlabel('Tempo (t)')
plt.ylabel('Posição (y)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('amortecimentos.png')