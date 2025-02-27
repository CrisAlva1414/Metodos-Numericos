import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definimos la función y su derivada
def f(x):
    return np.cos(x) - x

def df(x):
    return -np.sin(x) - 1

# Método de Newton-Raphson
def newton_raphson(x0, max_iter=10, tol=1e-6):
    x_vals = [x0]
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < tol:
            break
            
        # Calcular el siguiente punto usando la fórmula de Newton-Raphson
        x_next = x - fx / dfx
        x_vals.append(x_next)
        
        if abs(x_next - x) < tol:
            break
            
        x = x_next
        
    return x_vals

# Generar el gráfico
def create_visualization(x0=2.0):
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generar puntos para graficar la función
    x_range = np.linspace(-1, 3, 1000)
    y_range = f(x_range)
    
    # Graficar la función
    ax.plot(x_range, y_range, 'b-', label='f(x) = x³ - 2x - 5')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Configurar límites y etiquetas
    ax.set_xlim(-1, 3)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Método de Newton-Raphson')
    ax.legend()
    
    # Obtener las iteraciones
    iterations = newton_raphson(x0)
    
    # Lista para almacenar elementos gráficos
    tangent_line = None
    point_on_curve = None
    point_on_x = None
    vertical_line = None
    annotation = None
    
    # Función para la animación
    def update(frame):
        nonlocal tangent_line, point_on_curve, point_on_x, vertical_line, annotation
        
        # Limpiar elementos anteriores
        if tangent_line:
            tangent_line.remove()
        if point_on_curve:
            point_on_curve.remove()
        if point_on_x:
            point_on_x.remove()
        if vertical_line:
            vertical_line.remove()
        if annotation:
            annotation.remove()
        
        if frame < len(iterations):
            x_current = iterations[frame]
            y_current = f(x_current)
            slope = df(x_current)
            
            # Ecuación de la recta tangente: y - y_current = slope * (x - x_current)
            # y = slope * (x - x_current) + y_current
            x_tangent = np.linspace(x_current - 1, x_current + 1, 100)
            y_tangent = slope * (x_tangent - x_current) + y_current
            
            # Graficar la recta tangente
            tangent_line, = ax.plot(x_tangent, y_tangent, 'g-', label='Tangente')
            
            # Punto en la curva
            point_on_curve, = ax.plot(x_current, y_current, 'ro', markersize=8)
            
            # Siguiente punto en el eje x
            if frame < len(iterations) - 1:
                x_next = iterations[frame + 1]
                point_on_x, = ax.plot(x_next, 0, 'go', markersize=8)
                
                # Línea vertical desde el eje x hasta la curva
                vertical_line, = ax.plot([x_next, x_next], [0, f(x_next)], 'g--')
                
                # Añadir anotación
                annotation = ax.annotate(f'Iteración {frame+1}: x = {x_current:.6f}\nPróximo x = {x_next:.6f}',
                                        xy=(x_current, y_current),
                                        xytext=(x_current + 0.1, y_current + 1),
                                        arrowprops=dict(arrowstyle='->'))
            else:
                annotation = ax.annotate(f'Convergencia: x = {x_current:.6f}',
                                        xy=(x_current, y_current),
                                        xytext=(x_current + 0.1, y_current + 1),
                                        arrowprops=dict(arrowstyle='->'))
        
        return []
    
    # Crear la animación
    ani = FuncAnimation(fig, update, frames=len(iterations) + 1, blit=True, interval=1000)
    
    plt.tight_layout()
    plt.show()
    
    return iterations

# Ejecutar la visualización con un valor inicial x0 = 2.0
create_visualization(2.0)