import numpy
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# pos_p = neg_p = 0.5, alpha = 2
# we test 2-d case

dimension = 2
pos_p = 0.5
neg_p = 1 - pos_p
sample_N = 100000
H_1_size = 100  # divide 2pi
H_2_size = 100  # divide 2pi
eps = 0.01
delta = 0.01
constant_labels = 10

random_seed = 1
round_point_approx = 0.000001


def generate_H_v(H_v_size):
    '''
    :param H_v_size: size of hypothesis set H_v
    :return: hypothesis set H_v

    each hypothesis is represented as a d-dimensional vector
    H_i : numpy array of size: H_i_size * dimension
    '''

    H_v = numpy.ones(shape=(H_v_size, dimension))
    H_v_polar = numpy.zeros(shape=(H_v_size, dimension - 1))
    # Generate hypothesis by rotating unit vector in polar coordinates
    for i in range(dimension - 2):
        H_v_polar[:,i] = np.linspace(0.0, np.pi, H_v_size)

    H_v_polar[:,dimension-1] = np.linspace(0.0, 2*np.pi, H_v_size)

    H_v[:,0] = np.cos(H_v_polar[:,0])
    sin_polar = np.ones(shape=(H_v_size, 1))
    for i in range(1, dimension-1):
        sin_polar *= np.sin(H_v_polar[:,i-1])
        H_v[:,i] = sin_polar * np.cos(H_v_polar[:,i])

    H_v[:,-1] = sin_polar * np.sin(H_v_polar[:,-1])

    del H_v_polar

    return H_v


def generate_hypotheses():
    H_1 = generate_H_v(H_1_size)
    H_2 = generate_H_v(H_2_size)

    # Generate vectors by polar coordinates
    plt.plot()


def choose_concept(H):
    return H

def random_ball():
    # generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(sample_N, dimension))
    random_directions = normalize(random_directions, axis=1, norm='l2')

    return random_directions

def algorithm_1(X, H, eps, delta):
    return H

def test_accuracy(h_1, h_2, X):
    return ...

def main():
    H = generate_hypotheses()
    # c_1, c_2 = choose_concept(H)
    # X = random_ball()
    # # beta = min{alpha*eps/2, eps}
    # h_1,h_2 = algorithm_1(X, H, eps, delta)
    # test_accuracy(h_1, h_2, X)


if __name__ == '__main__':
    main()