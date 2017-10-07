import numpy as np


def gradientDescent(cost_function, gradient_func, point, max_iter=1000, tresh=0.0001,
                    step_type="fixed", step_size=0.0001, print_val=False):
    counter = 0
    list_coef = [point]
    prev_point = point

    while counter < max_iter:

        gradient = gradient_func(point)
        if step_type == "fixed":
            point = point - step_size * gradient

        elif step_type == "golden":
            point = point - goldenStep(cost_function, gradient_func, point) * gradient

        if np.linalg.norm(prev_point-point) < tresh:
            break

        list_coef.append(point)
        counter = counter + 1
        prev_point = point

    if print_val:
        print("steps: " + str(counter))

    return point, np.array(list_coef)


def goldenStep(function, gradient, point):
    def optimizer(s): return function(point - s * gradient(point))

    return goldenSearch(optimizer)


def goldenSearch(function, a=0, b=3, tresh=0.000001):
    golden_ratio = 0.618034

    # Define initial length for search
    length = b - a

    lambda_1 = a + golden_ratio ** 2 * length
    lambda_2 = a + golden_ratio * length

    while length > tresh:

        if function(lambda_1) > function(lambda_2):
            a = lambda_1
            lambda_1 = lambda_2
            length = b - a
            lambda_2 = a + golden_ratio * length
        else:
            b = lambda_2
            lambda_2 = lambda_1
            length = b - a
            lambda_1 = a + golden_ratio ** 2 * length

    return (b + a) / 2.0
