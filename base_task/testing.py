import numpy as np

from scipy.optimize import minimize


def rosenbrock(point):
    return (1 - point[0]) ** 2 + 100 * (point[1] - point[0] ** 2) ** 2


def quadratic(point):
    return (point[0] - 2) ** 2 + (point[1] + 3) ** 2


def rosenbrock_grad(point):
    dfdx0 = -2 * (1 - point[0]) - 400 * point[0] * (point[1] - point[0] ** 2)
    dfdx1 = 2 * 100 * (point[1] - point[0] ** 2)
    return np.array([dfdx0, dfdx1])


def quadratic_grad(point):
    dfdx0 = 2 * (point[0] - 2)
    dfdx1 = 2 * (point[1] + 3)
    return np.array([dfdx0, dfdx1])


def gradient_descent(grad_function, start_point, learning_rate=0.001, tolerance=1e-6, max_iterations=1000000,
                     max_evals=1000000):
    point = np.array(start_point, dtype=float)
    evals = 0
    it = 0
    for _ in range(max_iterations):
        grad = grad_function(point)
        point_new = point - learning_rate * grad

        '''if np.linalg.norm(point_new - point) < tolerance or point_new[0] == np.nan:  # check
            break'''
        if grad_function == rosenbrock_grad:
            if np.linalg.norm(point_new - np.array([1, 1])) < tolerance:
                break
        if grad_function == quadratic:
            if np.linalg.norm(point_new - np.array([2, -3])) < tolerance:
                break
        point = point_new
        evals += 1
        it += 1
        if evals >= max_evals:
            break
    print("gd-iterations:" + str(it))
    print("gd-evaluations:" + str(evals))
    return point


def gradient_descent_dichotomy(function, grad_function, start_point, tolerance=1e-6, max_iterations=10000000,
                               max_evals=3000000):
    point = np.array(start_point, dtype=float)
    evals = 0
    it = 0
    for _ in range(max_iterations):
        it += 1
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
            evals += 3
            if evals >= max_evals:
                break
        alpha_optimal = (alpha_left + alpha_right) / 2
        point_new = point - alpha_optimal * grad
        if function == rosenbrock:
            if np.linalg.norm(point_new - np.array([1, 1])) < tolerance:
                break
        if function == quadratic:
            if np.linalg.norm(point_new - np.array([2, -3])) < tolerance:
                break

        '''if np.linalg.norm(point_new - point) < tolerance:
            break'''
        point = point_new

    print("dich-iterations:" + str(it))
    print("dich-evaluations:" + str(evals))
    return point


def nelder_mead(function, start_point, precision):
    options = {
        'xatol': precision,
        'maxiter': 1000000,
        'maxfev': 300000
    }
    result = minimize(function, start_point, method='Nelder-Mead', options=options)
    return result.x


'''#start_point_rosenbrock = np.array([1, 1])
#start_point_quadratic = np.array([2, -3])
start_point_rosenbrock = np.array(st)
start_point_quadratic = np.array(st)

results = {}
print("----------POINTS TOLERANCE/PRECISION----------")
for required_tolerance in [1e-3, 1e-4, 1e-5, 1e-6]:
    print("=========================================================")
    #print("required tolerance:" + str(required_tolerance))
    print("required precision:" + str(required_tolerance))
    print()
    print("for rosenbrock:")
    gd_rosenbrock = gradient_descent(rosenbrock_grad, start_point_rosenbrock, 0.001, required_tolerance)
    gd_rosenbrock_dichotomy = gradient_descent_dichotomy(rosenbrock, rosenbrock_grad, start_point_rosenbrock,
                                                         required_tolerance)
    nm_rosenbrock = nelder_mead(rosenbrock, start_point_rosenbrock, required_tolerance)
    print()
    print("for quadratic:")
    gd_quadratic = gradient_descent(quadratic_grad, start_point_quadratic, 0.001, required_tolerance)
    gd_quadratic_dichotomy = gradient_descent_dichotomy(quadratic, quadratic_grad, start_point_quadratic,
                                                        required_tolerance)
    nm_quadratic = nelder_mead(quadratic, start_point_quadratic, required_tolerance)
    print()
    results["rosenbrock"] = {
        "gd": gd_rosenbrock,
        "gd_dichotomy": gd_rosenbrock_dichotomy,
        "nm": nm_rosenbrock
    }

    results["quadratic"] = {
        "gd": gd_quadratic,
        "gd_dichotomy": gd_quadratic_dichotomy,
        "nm": nm_quadratic
    }
    for key in results:
        print(str(key) + "values:")
        for method in results[key]:
            print(method, *results[key][method])
        print()

print("----------STARTING POINTS COMPARING----------")
for d in [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]:
        print("=========================================================")
        print("start rosenbrock:", str(start_point_rosenbrock[0] + d) + "," + str(start_point_rosenbrock[1] + d))
        print()
        print("start quadratic:", str(start_point_quadratic[0] + d) + "," + str(start_point_quadratic[1] + d))
        print()
        gd_rosenbrock = gradient_descent(rosenbrock_grad, start_point_rosenbrock + d, 0.001, 1e-4)
        gd_rosenbrock_dichotomy = gradient_descent_dichotomy(rosenbrock, rosenbrock_grad, start_point_rosenbrock + d,
                                                             1e-4)
        nm_rosenbrock = nelder_mead(rosenbrock, start_point_rosenbrock + d)
        gd_quadratic = gradient_descent(quadratic_grad, start_point_quadratic + d, 0.001, 1e-4)
        gd_quadratic_dichotomy = gradient_descent_dichotomy(quadratic, quadratic_grad, start_point_quadratic + d, 1e-4)
        nm_quadratic = nelder_mead(quadratic, start_point_quadratic + d)
        results["rosenbrock"] = {
            "gd": gd_rosenbrock,
            "gd_dichotomy": gd_rosenbrock_dichotomy,
            "nm": nm_rosenbrock
        }

        results["quadratic"] = {
            "gd": gd_quadratic,
            "gd_dichotomy": gd_quadratic_dichotomy,
            "nm": nm_quadratic
        }
        for key in results:
            print(str(key) + " values:")
            for method in results[key]:
                print(method, *results[key][method])
            print()'''




