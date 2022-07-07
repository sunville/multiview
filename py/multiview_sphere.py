import numpy
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d

# pos_p = neg_p = 0.5, alpha = 2
# we test 2-d case

dimension = 2
pos_p = 0.5
neg_p = 1 - pos_p
alpha = 2
sample_N = 1000000
H_1_delta = 1000  # divide 2pi
H_2_delta = 1000  # divide 2pi
eps = 0.01
confidence = 0.99
budget = 10
test = False
if test == False:
    random_seed = -1
else:
    random_seed = 1234

agnostic = False
if agnostic == False:
    agnostic_flip_p = 0
else:
    agnostic_flip_p = 0.1
round_point_approx = 0.0000001


def cartesian_product_transpose(arrays):
    broadcastable = numpy.ix_(*arrays)
    broadcasted = numpy.broadcast_arrays(*broadcastable)
    rows, cols = numpy.prod(broadcasted[0].shape), len(broadcasted)
    dtype = numpy.result_type(*arrays)
    out = numpy.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def generate_H_v(H_v_delta):
    '''
    :param H_v_size: size of hypothesis set H_v
    :return: hypothesis set H_v : shape=(H_v_delta**(dimension - 1) ,dimension)

    each hypothesis is represented as a d-dimensional vector
    H_i : numpy array of size: H_i_size * dimension
    '''

    H_v = numpy.ones(shape=(H_v_delta**(dimension-1), dimension))
    H_v_polar = numpy.zeros(shape=(H_v_delta, dimension - 1))

    # Generate hypothesis by rotating unit vector in polar coordinates
    for i in range(dimension - 2):
        H_v_polar[:, i] = np.linspace(0.0, np.pi, H_v_delta)

    H_v_polar[:, -1] = np.linspace(0.0, 2*np.pi, H_v_delta)
    H_v_polar = np.array([H_v_polar[:, i] for i in range(dimension - 1)])

    H_v_polar = cartesian_product_transpose(H_v_polar)

    H_v[:, 0] = np.cos(H_v_polar[:, 0])
    sin_polar = np.sin(H_v_polar[:, 0])

    for i in range(1, dimension-1):
        H_v[:, i] = sin_polar * np.cos(H_v_polar[:, i])
        sin_polar *= np.sin(H_v_polar[:, i])

    H_v[:, -1] = sin_polar

    # del H_v_polar

    # ax = plt.axes(projection='3d')
    # xline = H_v[:, 0]
    # yline = H_v[:, 1]
    # zline = H_v[:, 2]
    # ax.scatter3D(xline,yline,zline, cmap='Greens')
    # # plt.plot(xline, yline, 'bo')
    # plt.show()
    return H_v


def generate_hypotheses(H_1_size, H_2_size):
    '''

    :param H_1_size: size of hypothesis set H_1
    :param H_2_size: size of hypothesis set H_2
    :return: shape=(H_1_size * H_2_size, 2)
             All possible pairs of H_1 X H_2.
             Cartesian product of [1,..,H_1_size] X [1,...,H_2_size]
    '''

    H_1_indices = np.array(np.arange(H_1_size))
    H_2_indices = np.array(np.arange(H_2_size))
    H_indices = cartesian_product_transpose([H_1_indices, H_2_indices])
    return H_indices


def choose_concept(H_indices, rng):

    h_index = rng.integers(H_indices.shape[0])
    c_1_index, c_2_index = H_indices[h_index, :]

    return c_1_index, c_2_index


def generate_labels(rng):
    '''

    :param rng: Generators
    :return: labels of all instances.
             shape: (1,)
             labels space = {-1, 1}
    '''
    labels = rng.uniform(0, 1, size=sample_N)
    labels[labels <= neg_p] = -1
    labels[labels > neg_p] = 1

    return labels


def generate_instance(c_v, label, rng):

    if random_seed > 0:
        random_directions = rng.standard_normal(size=(1, dimension))
    else:
        random_directions = np.random.normal(size=(1, dimension))

    if label > 0:
        while np.vdot(random_directions, c_v) < -round_point_approx:
            if test:
                random_directions = rng.standard_normal(size=(1, dimension))
            else:
                random_directions = np.random.normal(size=(1, dimension))
    else:
        while np.vdot(random_directions, c_v) > round_point_approx:
            if test:
                random_directions = rng.standard_normal(size=(1, dimension))
            else:
                random_directions = np.random.normal(size=(1, dimension))

    random_directions = normalize(random_directions, axis=1, norm='l2')
    return random_directions


def random_ball(c_v, labels, rng, figname):
    # generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    samples = np.empty(shape=(sample_N, dimension))

    for index in range(sample_N):
        if labels[index] == 1:
            samples[index, :] = generate_instance(c_v, 1, rng)
        else:
            samples[index, :] = generate_instance(c_v, -1, rng)

    if figname == "./X_1_concept":
        plt.figure(1)
    else:
        plt.figure(2)

    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=ListedColormap(['blue', 'red']))
    plt.plot([1.1, -1.1], [-1.1 * c_v[0] / c_v[1], 1.1 * c_v[0] / c_v[1]], color='black')
    plt.savefig(figname)

    return samples


def composite_h(h_1, h_2, x_1, x_2):
    if np.sign(np.vdot(h_1, x_1)) == np.sign(np.vdot(h_2, x_2)):
        return np.sign(np.vdot(h_1, x_1))
    else:
        if np.random.uniform(0, 1) > 0.5:
            return 1
        else:
            return -1


def algorithm_1(X_1, X_2, labels, H_1, H_2, H_indices, beta, delta, rng):
    '''
    :param X_1: shape=(sample_N, dimension + 1) view 1
    :param X_2: shape=(sample_N, dimension + 1) view 2
    :param H_indices: shape=(H_1_size * H_2_size, 2)
    :param beta:
    :param delta:
    :return: hypothesis indices of h_1 and h_2
    '''

    H_s = np.full(H_indices.shape[0], True)
    unlabeled_samples_num = np.log(H_indices.shape[0]) / beta * np.log(1/(beta*delta)) / 100
    unlabeled_samples_num = int(unlabeled_samples_num)

    print("The number of unlabeled samples are {}.".format(unlabeled_samples_num))

    unlabeled_samples_indices = rng.integers(X_1.shape[0], size=unlabeled_samples_num)

    for index in range(H_indices.shape[0]):
        h_1, h_2 = H_1[H_indices[index, 0], :], H_2[H_indices[index, 1], :]
        h_1_ret = np.sign(np.matmul(X_1[unlabeled_samples_indices, :], h_1.T))
        h_2_ret = np.sign(np.matmul(X_2[unlabeled_samples_indices, :], h_2.T))

        if np.array_equal(h_1_ret, h_2_ret):
            ...
        else:
            H_s[index] = False

    # store the index of survived hypothesis pair in H_indices
    H_s = np.nonzero(H_s == True)[0]

    print("The size of H_s is {}".format(H_s.shape[0]))

    labeled_samples_indices = rng.integers(X_1.shape[0], size=budget)

    requested_labels = labels[labeled_samples_indices]
    labeled_x_1 = X_1[labeled_samples_indices, :]
    labeled_x_2 = X_2[labeled_samples_indices, :]
    accuracy_list = np.zeros(shape=H_s.shape[0])

    for i, h_index in enumerate(H_s):
        h_1, h_2 = H_1[H_indices[h_index, 0], :], H_2[H_indices[h_index, 1], :]
        error = 0.0
        for label_index in range(budget):
            if requested_labels[label_index] == composite_h(h_1, h_2, labeled_x_1[label_index, :], labeled_x_2[label_index, :]):
                ...
            else:
                error += 1
        accuracy_list[i] = error/budget

    best_hypothesis_indices = np.nonzero(accuracy_list == np.amin(accuracy_list))[0]
    chosen_hypothesis_index = np.random.choice(best_hypothesis_indices)

    h_1 = H_1[H_indices[H_s[chosen_hypothesis_index], 0], :]
    h_2 = H_2[H_indices[H_s[chosen_hypothesis_index], 1], :]

    return h_1, h_2


def test_accuracy(h_1, h_2, X_1, X_2, labels):
    h_1_ret = np.sign(np.matmul(X_1, h_1.T))
    h_2_ret = np.sign(np.matmul(X_2, h_2.T))
    disagreement_region = np.nonzero(h_1_ret - h_2_ret)[0]
    flip_ret = np.sign(np.random.uniform(-1, 1, disagreement_region.shape[0]))

    h_1_ret[disagreement_region] = flip_ret
    error = (np.nonzero(h_1_ret - labels)[0]).shape[0]

    error /= labels.shape[0]

    print("Error of chosen h_1: {}, h_2: {}, is {}".format(h_1, h_2, error))

    plt.figure(3)
    plt.scatter(X_1[:, 0], X_1[:, 1], c=labels, cmap=ListedColormap(['blue', 'red']))
    plt.plot([1.1, -1.1], [-1.1*h_1[0]/h_1[1], 1.1*h_1[0]/h_1[1]], color='black')
    plt.savefig("X_1_h_1")

    plt.figure(4)
    plt.scatter(X_2[:, 0], X_2[:, 1], c=labels, cmap=ListedColormap(['blue', 'red']))
    plt.plot([1.1, -1.1], [-1.1*h_2[0]/h_2[1], 1.1*h_2[0]/h_2[1]], color='black')
    plt.savefig("X_2_h_2")


def main():
    H_1 = generate_H_v(H_1_delta)
    H_2 = generate_H_v(H_2_delta)
    H_indices = generate_hypotheses(H_1.shape[0], H_2.shape[0])

    if random_seed > 0:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()

    c_1_index, c_2_index = choose_concept(H_indices, rng)
    labels = generate_labels(rng)
    X_1 = random_ball(H_1[c_1_index, :], labels, rng, "./X_1_concept")
    X_2 = random_ball(H_2[c_2_index, :], labels, rng, "./X_2_concept")
    beta = min(alpha*eps/2, eps)
    h_1, h_2 = algorithm_1(X_1, X_2, labels, H_1, H_2, H_indices, beta, 1-confidence, rng)
    test_accuracy(h_1, h_2, X_1, X_2, labels)


if __name__ == '__main__':
    main()