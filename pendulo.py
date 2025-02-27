import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del péndulo
g = 9.81  # aceleración gravitacional (m/s^2)
L = 1.0   # longitud del péndulo (m)
m = 1.0   # masa (kg)
b = 0.1   # coeficiente de amortiguamiento

# Condiciones iniciales [theta, omega]
theta0 = np.pi/3  # ángulo inicial (radianes)
omega0 = 0.0      # velocidad angular inicial (rad/s)
state0 = np.array([theta0, omega0])

# Parámetros de simulación
t0 = 0.0
tf = 20.0
dt = 0.01
t = np.arange(t0, tf, dt)

# Función para derivadas del péndulo
def pendulum_derivatives(state, t):
    theta, omega = state
    
    # Ecuaciones del movimiento para un péndulo
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta) - (b/m) * omega
    
    return np.array([dtheta_dt, domega_dt])

# Implementación del método Runge-Kutta de 4to orden
def runge_kutta4(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + dt/2 * k1, t + dt/2)
    k3 = f(state + dt/2 * k2, t + dt/2)
    k4 = f(state + dt * k3, t + dt)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Simular el péndulo
def simulate_pendulum():
    # Arreglos para almacenar resultados
    states = np.zeros((len(t), 2))
    states[0] = state0
    
    # Resolver con RK4
    for i in range(1, len(t)):
        states[i] = runge_kutta4(pendulum_derivatives, states[i-1], t[i-1], dt)
    
    return states

# Ejecutar la simulación
states = simulate_pendulum()
theta = states[:, 0]
omega = states[:, 1]

# Convertir coordenadas polares a cartesianas para visualización
x = L * np.sin(theta)
y = -L * np.cos(theta)

# Visualizar resultados
plt.figure(figsize=(12, 6))

# Gráfica del ángulo vs tiempo
plt.subplot(121)
plt.plot(t, theta, 'b-', label='Ángulo θ')
plt.grid(True)
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Ángulo del péndulo vs tiempo')
plt.legend()

# Gráfica de la velocidad angular vs tiempo
plt.subplot(122)
plt.plot(t, omega, 'r-', label='Velocidad angular ω')
plt.grid(True)
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad angular (rad/s)')
plt.title('Velocidad angular vs tiempo')
plt.legend()

plt.tight_layout()
plt.show()

# Crear animación
def create_animation():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2*L, 1.2*L)
    ax.set_ylim(-1.2*L, 1.2*L)
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'Tiempo = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        # Solo animamos cada 5 frames para hacer la animación más rápida
        i = i * 5
        if i >= len(t):
            i = len(t) - 1
            
        pendulum_x = [0, x[i]]
        pendulum_y = [0, y[i]]
        line.set_data(pendulum_x, pendulum_y)
        time_text.set_text(time_template % t[i])
        return line, time_text
    
    ani = FuncAnimation(fig, animate, frames=len(t)//5, 
                        interval=50, blit=True, init_func=init)
    plt.title('Simulación del Péndulo')
    plt.xlabel('Posición x')
    plt.ylabel('Posición y')
    
    return ani

# Descomentar para crear y guardar la animación
ani = create_animation()
ani.save('pendulum_simulation.gif', writer='pillow', fps=20)
plt.show()

print("Simulación completa. Para ver la animación, descomente las últimas líneas del código.")