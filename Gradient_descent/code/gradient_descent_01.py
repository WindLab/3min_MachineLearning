import matplotlib.pyplot as plt
import numpy as np
import pdb
from numpy import *
import time

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 2]
        y = points[i, 3]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gen_plot(points, b, m, file_name):
    _, _, x, y = points.T
    y2 = map(lambda e: e*m + b, x)
    plt.plot(x, y2, 'k-2')
    plt.scatter(x, y)
    plt.ylabel('Blood pressure')
    plt.xlabel('Age')
    plt.savefig(file_name + "_regression")

def gen_origin(points):
    _, _, x, y = points.T
    plt.scatter(x, y)
    plt.ylabel('Blood pressure')
    plt.xlabel('Age')
    plt.savefig("raw_plot")

def gen_mse_plot(error):
    plt.plot(error)
    plt.ylabel('mse')
    plt.xlabel('number of iterations')
    plt.savefig("msechange")

def gen_contour_plot(points):
    xlist = np.linspace(92, 94.0, 1000)
    ylist = np.linspace(0.5, 1.5, 1000)
    X, Y = np.meshgrid(xlist, ylist)
    Z = compute_error_for_line_given_points(X, Y, array(points))
    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.savefig("contourplot")

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 2]
        y = points[i, 3]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    error = compute_error_for_line_given_points(new_b, new_m, array(points))
    return [new_b, new_m, error]

def gradient_descent_runner_fixnum(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    error = np.zeros(num_iterations)
    for i in range(num_iterations):
        b, m, error[i]= step_gradient(b, m, array(points), learning_rate)
    return [b, m, error]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate):
    b = starting_b
    m = starting_m
    error = []
    temp_error = compute_error_for_line_given_points(b, m, array(points))
    while temp_error > 150:
        # how to naturally push this to its limit
        # could contour map help us in this case ?
        b, m, temp_error= step_gradient(b, m, array(points), learning_rate)
        error.append(temp_error)

    return [b, m, np.array(error)]

if __name__ == '__main__':
    points = np.genfromtxt("dataset/age_blood_pressure_clean.txt", delimiter="")
    gen_origin(points)

    num_iterations = 5
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."

    start_time = time.time()

    [b, m, error] = gradient_descent_runner_fixnum(points, initial_b, initial_m, learning_rate, num_iterations)

    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))
    print("--- %s seconds ---" % (time.time() - start_time))
    gen_plot(points, b, m, str(num_iterations))

    num_iterations = 100000
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."

    start_time = time.time()

    [b, m, error] = gradient_descent_runner_fixnum(points, initial_b, initial_m, learning_rate, num_iterations)

    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))
    print("--- %s seconds ---" % (time.time() - start_time))
    gen_plot(points, b, m, str(num_iterations))

    # [b, m, error] = gradient_descent_runner(points, initial_b, initial_m, learning_rate)
    # gen_mse_plot(error)
    # gen_contour_plot(points)
