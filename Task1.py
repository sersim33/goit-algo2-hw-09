import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Функція Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)

# Hill Climbing з відстеженням шляху
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    num_dimensions = len(bounds)
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)]
    current_value = func(current_solution)
    path = [current_solution[:]]

    for _ in range(iterations):
        next_solution = current_solution[:]
        index = random.randint(0, num_dimensions - 1)
        next_solution[index] += np.random.uniform(-epsilon, epsilon)
        next_value = func(next_solution)

        if next_value < current_value:
            current_solution, current_value = next_solution, next_value
            path.append(current_solution[:])

        if abs(next_value - current_value) < epsilon:
            break

    return current_solution, current_value, path

# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    num_dimensions = len(bounds)
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)]
    current_value = func(current_solution)
    path = [current_solution[:]]

    for _ in range(iterations):
        next_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)]
        next_value = func(next_solution)

        if next_value < current_value:
            current_solution, current_value = next_solution, next_value
            path.append(current_solution[:])

        if abs(next_value - current_value) < epsilon:
            break

    return current_solution, current_value, path

# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    num_dimensions = len(bounds)
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_dimensions)]
    current_value = func(current_solution)
    path = [current_solution[:]]

    for _ in range(iterations):
        next_solution = current_solution[:]
        index = random.randint(0, num_dimensions - 1)
        next_solution[index] += np.random.uniform(-epsilon, epsilon)
        next_value = func(next_solution)

        if next_value < current_value or math.exp((current_value - next_value) / temp) > np.random.rand():
            current_solution, current_value = next_solution, next_value
            path.append(current_solution[:])

        temp *= cooling_rate
        if temp < epsilon or abs(next_value - current_value) < epsilon:
            break

    return current_solution, current_value, path

# Функція для візуалізації
def plot_paths(bounds, paths, labels, title):
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=30, cmap='viridis')
    colors = ['r', 'g', 'b']

    for path, label, color in zip(paths, labels, colors):
        px, py = zip(*path)
        plt.plot(px, py, marker='o', label=label, color=color, linewidth=2)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Запуск
if __name__ == "__main__":
    bounds = [(-5, 5), (-5, 5)]

    hc_solution, hc_value, hc_path = hill_climbing(sphere_function, bounds)
    rls_solution, rls_value, rls_path = random_local_search(sphere_function, bounds)
    sa_solution, sa_value, sa_path = simulated_annealing(sphere_function, bounds)

    print("Hill Climbing:", hc_solution, "->", hc_value)
    print("Random Local Search:", rls_solution, "->", rls_value)
    print("Simulated Annealing:", sa_solution, "->", sa_value)

    plot_paths(bounds,
               [hc_path, rls_path, sa_path],
               ["Hill Climbing", "Random Local Search", "Simulated Annealing"],
               "Порівняння шляхів оптимізації")
