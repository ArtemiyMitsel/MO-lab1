from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def quadratic(point):
    return (point[0] - 2) ** 2 + (point[1] + 3) ** 2


def quadratic_grad(point):
    dfdx0 = 2 * (point[0] - 2)
    dfdx1 = 2 * (point[1] + 3)
    return np.array([dfdx0, dfdx1])


def gradient_descent_with_trajectory(grad_function, start_point, learning_rate=0.001, tolerance=1e-6,
                                     max_iterations=1000000):
    point = np.array(start_point, dtype=float)
    trajectory = [point.copy()]
    for _ in range(max_iterations):
        grad = grad_function(point)
        point_new = point - learning_rate * grad
        if np.linalg.norm(point_new - point) < tolerance:
            break
        point = point_new
        trajectory.append(point.copy())
    return point, trajectory


start_point_quadratic = [0.5, -2]  # Example start point
final_point, trajectory = gradient_descent_with_trajectory(quadratic_grad, start_point_quadratic)

final_point, len(trajectory)


def gradient_descent_dichotomy_with_trajectory(function, grad_function, start_point, tolerance=1e-6,
                                               max_iterations=10000000):
    point = np.array(start_point, dtype=float)
    trajectory = [point.copy()]
    for _ in range(max_iterations):
        grad = grad_function(point)

        def f_new(alpha):
            return function(point - alpha * grad)

        alpha_left, alpha_right = 0, 1
        while alpha_right - alpha_left > tolerance:
            alpha_mid = (alpha_left + alpha_right) / 2
            if f_new(alpha_mid - tolerance / 2) < f_new(alpha_mid + tolerance / 2):
                alpha_right = alpha_mid
            else:
                alpha_left = alpha_mid
        alpha_optimal = (alpha_left + alpha_right) / 2
        point_new = point - alpha_optimal * grad
        if np.linalg.norm(point_new - point) < tolerance:
            break
        point = point_new
        trajectory.append(point.copy())
    return point, trajectory


final_point_dich, trajectory_dich = gradient_descent_dichotomy_with_trajectory(quadratic, quadratic_grad,
                                                                               start_point_quadratic)

final_point_dich, len(trajectory_dich)


def nelder_mead_with_trajectory(function, start_point, precision=1e-6):
    trajectory = [np.array(start_point)]

    def callback(x):
        trajectory.append(x.copy())

    result = minimize(function, start_point, method='Nelder-Mead', options={'xatol': precision, 'disp': True},
                      callback=callback)
    return result.x, trajectory


final_point_nm, trajectory_nm = nelder_mead_with_trajectory(quadratic, start_point_quadratic)
final_point_nm, len(trajectory_nm)


def rosenbrock(point):
    return (1 - point[0]) ** 2 + 100 * (point[1] - point[0] ** 2) ** 2


def plot_contour_and_trajectory(function, grad_function, start_point, xlim, ylim, title_prefix, trajectories):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])

    # Plotting contour
    plt.figure(figsize=(18, 6))
    for i, (method_name, trajectory) in enumerate(trajectories.items(), 1):
        plt.subplot(1, 3, i)
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.plot(*zip(*trajectory), marker='o', color='r', markersize=4, linestyle='-', linewidth=1, label='Path')
        plt.scatter(*start_point, color='b', label='Start')
        plt.scatter(*trajectory[-1], color='g', label='End')
        plt.title(f'{title_prefix} - {method_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
    plt.tight_layout()
    if function == rosenbrock:
        plt.savefig("rosenbrock_traectories.png")
    else:
        plt.savefig("quad_traectories.png")


plot_contour_and_trajectory(
    quadratic, quadratic_grad, start_point_quadratic, [0, 3], [-5, -1],
    "Quadratic Function",
    {"Gradient Descent": trajectory, "Dichotomy": trajectory_dich, "Nelder-Mead": trajectory_nm}
)


def rosenbrock_grad(point):
    dfdx0 = -2 * (1 - point[0]) - 400 * point[0] * (point[1] - point[0] ** 2)
    dfdx1 = 2 * 100 * (point[1] - point[0] ** 2)
    return np.array([dfdx0, dfdx1])


start_point_rosenbrock = [-1, 2]

final_point_rosen_gd, trajectory_rosen_gd = gradient_descent_with_trajectory(rosenbrock_grad, start_point_rosenbrock)

final_point_rosen_dich, trajectory_rosen_dich = gradient_descent_dichotomy_with_trajectory(rosenbrock, rosenbrock_grad,
                                                                                           start_point_rosenbrock)

final_point_rosen_nm, trajectory_rosen_nm = nelder_mead_with_trajectory(rosenbrock, start_point_rosenbrock)

plot_contour_and_trajectory(
    rosenbrock, rosenbrock_grad, start_point_rosenbrock, [-2, 2], [-1, 3],
    "Rosenbrock Function",
    {"Gradient Descent": trajectory_rosen_gd, "Dichotomy": trajectory_rosen_dich, "Nelder-Mead": trajectory_rosen_nm}
)
