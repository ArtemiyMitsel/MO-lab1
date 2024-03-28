import random
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# функции и градиенты
def quadratic(point):
    return (point[0] - 2) ** 2 + (point[1] + 3) ** 2

def quadratic_grad(point):
    dfdx0 = 2 * (point[0] - 2)
    dfdx1 = 2 * (point[1] + 3)
    return np.array([dfdx0, dfdx1])

def rosenbrock(point):
    return (1 - point[0]) ** 2 + 100 * (point[1] - point[0] ** 2) ** 2

def rosenbrock_grad(point):
    dfdx0 = -2 * (1 - point[0]) - 400 * point[0] * (point[1] - point[0] ** 2)
    dfdx1 = 200 * (point[1] - point[0] ** 2)
    return np.array([dfdx0, dfdx1])

# Методы оптимизации
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

def nelder_mead_with_trajectory(function, start_point, precision=1e-6):
    trajectory = [np.array(start_point)]

    def callback(x):
        trajectory.append(x.copy())

    result = minimize(function, start_point, method='Nelder-Mead', options={'xatol': precision, 'disp': True},
                      callback=callback)
    return result.x, trajectory


# Доп. задание 1


def gradient_descent_golden_section_with_trajectory(function, grad_function, start_point, tolerance=1e-6,
                                                    max_iterations=100000):
    golden_ratio = (np.sqrt(5) - 1) / 2
    point = np.array(start_point, dtype=float)
    trajectory = [point.copy()]

    for _ in range(max_iterations):
        grad = grad_function(point)

        def f(alpha):
            return function(point - alpha * grad)

        a, b = 0, 1
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)
        while abs(c - d) > tolerance:
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - golden_ratio * (b - a)
            d = a + golden_ratio * (b - a)
        alpha_optimal = (b + a) / 2

        point_new = point - alpha_optimal * grad
        if np.linalg.norm(point_new - point) < tolerance:
            break
        point = point_new
        trajectory.append(point.copy())

    return point, trajectory

# Доп. задание 2


# Пункт 1
def extended_rosenbrock(x):
    return sum(100.0*(x[i+1]-x[i]**2.0)**2.0 + (1-x[i])**2.0 for i in range(len(x)-1))

def extended_rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400.0*x[0]*(x[1]-x[0]**2) - 2.0*(1-x[0])
    for i in range(1,len(x)-1):
        grad[i] = 200.0*(x[i]-x[i-1]**2) - 400.0*(x[i+1]-x[i]**2)-(2.0*(1-x[i]))
    grad[-1] = 200.0*(x[-1]-x[-2]**2)
    return grad

# Пункт 2

def goldstein_price(point):
    x, y = point[0], point[1]
    fact1a = (x + y + 1)**2
    fact1b = 19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2
    fact1 = 1 + fact1a*fact1b
    fact2a = (2*x - 3*y)**2
    fact2b = 18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2
    fact2 = 30 + fact2a*fact2b
    return fact1*fact2

def goldstein_price_grad(point):
    x, y = point[0], point[1]
    dfdx = ((x + y + 1)*(80*x**3 - 170*x**2 + 86*x + 116*y**2 + 4*y + 64) - (2*x - 3*y)*(64*x**3 - 276*x**2 + 384*x - 200*y**2 - 144*y - 24))/(4*(x**2 - 4*x*y + 4*y**2 + 1)**2)
    dfdy = ((x + y + 1)*(-12*x**2 + 16*x - 2*y**2 + 16) + (2*x - 3*y)*(-48*x**2 + 156*x - 222*y**2 - 58*y + 84))/(4*(x**2 - 4*x*y + 4*y**2 + 1)**2)
    return np.array([dfdx, dfdy])

def rastrigin(point):
    x, y  = point[0], point[1]
    return 10*2 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

def rastrigin_grad(point):
    x, y = point[0], point[1]
    dfdx = 2*x + 20*np.pi*x*np.sin(2*np.pi*x)
    dfdy = 2*y + 20*np.pi*y*np.sin(2*np.pi*y)
    return np.array([dfdx, dfdy])

# Пункт 3

def noisy_quadratic(point):
    noise = random.normalvariate(0, 0.05)
    ans = (point[0] - 2) ** 2 + (point[1] + 3) ** 2 + noise
    return ans

# Визуализация
def plot_contour_and_trajectory(function, grad_function, start_point, xlim, ylim, title_prefix, trajectories, imgName=None):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])

    # Plotting contour
    plt.figure(figsize=(24, 6))
    for i, (method_name, trajectory) in enumerate(trajectories.items(), 1):
        plt.subplot(1, 4, i)
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
        if imgName is None:
            plt.savefig("quad_traectories.png")
        else:
            plt.savefig(imgName)



if __name__ == "__main__":
    # Точки старта для различных функций
    start_point_quadratic = [0.5, -2]
    start_point_rosenbrock = [-1, 2]

    # Оптимизация квадратичной функции
    _, trajectory = gradient_descent_with_trajectory(quadratic_grad, start_point_quadratic)
    _, trajectory_quad_golden = gradient_descent_golden_section_with_trajectory(quadratic, quadratic_grad, start_point_quadratic)
    _, trajectory_dich = gradient_descent_dichotomy_with_trajectory(quadratic, quadratic_grad, start_point_quadratic)
    _, trajectory_quadratic_nm = nelder_mead_with_trajectory(quadratic, start_point_quadratic)

    plot_contour_and_trajectory(
        quadratic, quadratic_grad, start_point_quadratic, [0, 3], [-5, -1],
        "Quadratic Function",
        {"Gradient Descent": trajectory, "Dichotomy": trajectory_dich, "Nelder-Mead": trajectory_quadratic_nm, "Golden Section": trajectory_quad_golden }
    )

    # Оптимизация функции Розенброка
    _, trajectory_rosen_gd = gradient_descent_with_trajectory(rosenbrock_grad, start_point_rosenbrock)
    _, trajectory_rosen_golden = gradient_descent_golden_section_with_trajectory(rosenbrock, rosenbrock_grad, start_point_rosenbrock)
    _, trajectory_rosen_dich = gradient_descent_dichotomy_with_trajectory(rosenbrock, rosenbrock_grad, start_point_rosenbrock)
    _, trajectory_rosen_nm = nelder_mead_with_trajectory(rosenbrock, start_point_rosenbrock)

    plot_contour_and_trajectory(
        rosenbrock, rosenbrock_grad, start_point_rosenbrock, [-2, 2], [-1, 3],
        "Rosenbrock Function",
        {
            "Gradient Descent": trajectory_rosen_gd,
            "Dichotomy": trajectory_rosen_dich,
            "Nelder-Mead": trajectory_rosen_nm,
            "Golden Section": trajectory_rosen_golden
        }
    )

    # Доп. задание 2

    # Пункт 1
    for i in range(1, 5):
        start_point_extended = []
        for _ in range(i):
            start_point_extended.append(-1)
            start_point_extended.append(2)
        print(start_point_extended)

        final_point_ext_gd, trajectory_ext_gd = gradient_descent_with_trajectory(extended_rosenbrock_grad,
                                                                                 start_point_extended)
        final_point_ext_dich, trajectory_ext_dich = gradient_descent_dichotomy_with_trajectory(extended_rosenbrock,
                                                                                               extended_rosenbrock_grad,
                                                                                               start_point_extended)
        final_point_ext_nm, trajectory_ext_nm = nelder_mead_with_trajectory(extended_rosenbrock, start_point_extended)

        # Размерность датасетов
        print(len(trajectory_ext_gd), len(trajectory_ext_dich), len(trajectory_ext_nm))


    # Пункт 2

    # Начальная точка для плохо обусловленных функций
    start_point_bad = [0.5, -0.5]

    # Оптимизация с использованием функции Гольдштейна–Прайса
    final_point_gp_gd, trajectory_gp_gd = gradient_descent_with_trajectory(goldstein_price_grad, start_point_bad)
    final_point_gp_dich, trajectory_gp_dich = gradient_descent_dichotomy_with_trajectory(goldstein_price,
                                                                                         goldstein_price_grad,
                                                                                         start_point_bad)
    final_point_gp_nm, trajectory_gp_nm = nelder_mead_with_trajectory(goldstein_price, start_point_bad)
    plot_contour_and_trajectory(
        goldstein_price, goldstein_price_grad, start_point_bad, [-0.5, 2], [-1.5, -0.25],
        "Goldstein-Price Function",
        {"Gradient Descent": trajectory_gp_gd, "Dichotomy": trajectory_gp_dich, "Nelder-Mead": trajectory_gp_nm},
        "goldstein_price_traectories.png"
    )

    # Оптимизация с использованием функции Растригина
    final_point_r_gd, trajectory_r_gd = gradient_descent_with_trajectory(rastrigin_grad, start_point_bad)
    final_point_r_dich, trajectory_r_dich = gradient_descent_dichotomy_with_trajectory(rastrigin, rastrigin_grad,
                                                                                       start_point_bad)
    final_point_r_nm, trajectory_r_nm = nelder_mead_with_trajectory(rastrigin, start_point_bad)
    plot_contour_and_trajectory(
        rastrigin, rastrigin_grad, start_point_bad, [-0.5, 1.5], [-1.5, 0.5],
        "Rastrigin Function",
        {"Gradient Descent": trajectory_r_gd, "Dichotomy": trajectory_r_dich, "Nelder-Mead": trajectory_r_nm},
        "rastrigin_traectories.png"
    )

    # Пункт 3

    start_point_quadratic = [0.5, -2]

    final_point_nm, trajectory_nm = nelder_mead_with_trajectory(quadratic, start_point_quadratic)
    final_point_nm, len(trajectory_nm)
    final_point_nm, trajectory_nm_noisy = nelder_mead_with_trajectory(noisy_quadratic, start_point_quadratic)
    final_point_nm, len(trajectory_nm_noisy)

    plot_contour_and_trajectory(
        noisy_quadratic, None, start_point_quadratic, [0, 3], [-5, -1],
        "Quadratic Function",
        {"Nelder-Mead with Noisy": trajectory_nm_noisy, "Nelder-Mead": trajectory_nm},
        "noisy_quad_traectories.png"
    )


