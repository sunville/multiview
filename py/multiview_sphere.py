import numpy
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import argparse


parser = argparse.ArgumentParser(description='Multiview Learning on Unit Sphere.')
parser.add_argument('--d_1', type=int, nargs='?', default=2,
                    help='dimension of X_1')
parser.add_argument('--d_2', type=int, nargs='?', default=2,
                    help='dimension of X_2')
parser.add_argument('--pos_p', type=float, nargs='?', default=0.5,
                    help='Pr(y=1)')
parser.add_argument('--N', type=int, nargs='?', default=10000,
                    help='number of total samples')
parser.add_argument('--H_1_delta', type=int, nargs='?', default=100,
                    help='number of angles we will evenly cut on each freedom of spherical coordinate system in d_1-dimensional space')
parser.add_argument('--H_2_delta', type=int, nargs='?', default=100,
                    help='number of angles we will evenly cut on each freedom of spherical coordinate system in d_2-dimensional space')
parser.add_argument('--eps', type=float, nargs='?', default=0.01,
                    help='error tolerance')
parser.add_argument('--confidence', type=float, nargs='?', default=0.99,
                    help='confidence')
parser.add_argument('--budget', type=int, nargs='?',
                    help='number of labeled samples')
parser.add_argument('--flip_p', type=float, nargs='?', default=0,
                    help='probability of flipping the label after generating a sample')
parser.add_argument('--batch', type=int, nargs='?', default=0,
                    help='unlabeled_samples_batch')
parser.add_argument('--init', type=int, nargs='?', default=0,
                    help='start recording statistics after drawing such number of unlabeled samples')
parser.add_argument('--random_seed', type=int, nargs='?',
                    help='test mode')

opt = parser.parse_args()

print(opt)

dimension_1 = opt.d_1
dimension_2 = opt.d_2
pos_p = opt.pos_p
neg_p = 1 - pos_p
# alpha = 2
sample_N = opt.N
H_1_delta = opt.H_1_delta  # divide 2pi
H_2_delta = opt.H_2_delta  # divide 2pi
eps = opt.eps
confidence = opt.confidence
budget = opt.budget
flip_p = opt.flip_p
random_seed = opt.random_seed
unlabeled_samples_batch = opt.batch
unlabeled_samples_init = opt.init

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


def generate_H_v(H_v_delta, dimension):
    '''
    :param H_v_size: size of hypothesis set H_v
    :return: hypothesis set H_v : shape=(H_v_delta**(dimension - 1) ,dimension)

    each hypothesis is represented as a d-dimensional vector
    H_i : numpy array of size: H_i_size * dimension
    '''

    H_v = numpy.ones(shape=(H_v_delta ** (dimension - 1), dimension))
    H_v_polar = numpy.zeros(shape=(H_v_delta, dimension - 1))

    # Generate hypothesis by rotating unit vector in polar coordinates
    for i in range(dimension - 2):
        H_v_polar[:, i] = np.linspace(0.0, np.pi, H_v_delta)

    H_v_polar[:, -1] = np.linspace(0.0, 2 * np.pi, H_v_delta)
    H_v_polar = np.array([H_v_polar[:, i] for i in range(dimension - 1)])

    H_v_polar = cartesian_product_transpose(H_v_polar)

    H_v[:, 0] = np.cos(H_v_polar[:, 0])
    sin_polar = np.sin(H_v_polar[:, 0])

    for i in range(1, dimension - 1):
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


# def generate_hypotheses(H_1_size, H_2_size):
#     '''
#
#     :param H_1_size: size of hypothesis set H_1
#     :param H_2_size: size of hypothesis set H_2
#     :return: shape=(H_1_size * H_2_size, 2)
#              All possible pairs of H_1 X H_2.
#              Cartesian product of [1,..,H_1_size] X [1,...,H_2_size]
#     '''
#
#     H_1_indices = np.array(np.arange(H_1_size))
#     H_2_indices = np.array(np.arange(H_2_size))
#     H_indices = cartesian_product_transpose([H_1_indices, H_2_indices])
#     print(f'Total number of hypotheses: {H_1_size * H_2_size}')
#     return H_indices


def choose_concept(H_1_size, H_2_size, rng):
    '''
    :param H_1_size: size of H_1
    :param H_2_size: size of H_2
    :param rng: random number generator
    :return: index of c_1 and c_2
    '''

    c_1_index = rng.integers(H_1_size)
    c_2_index = rng.integers(H_2_size)

    return c_1_index, c_2_index


def generate_labels(N, pos_p, rng):
    '''

    :param N: number of samples
    :param pos_p: Pr(y=1)
    :param rng: random number generator
    :return: labels: 1d array
             pos_samples_num: number of positive samples
             neg_samples_num: number of negtive samples
    '''
    pos_samples_num = np.nonzero(rng.uniform(0, 1, size=N) < pos_p)[0].size
    labels = np.zeros(N)
    labels[0: pos_samples_num] = 1
    labels[pos_samples_num: N] = -1
    print('positive samples: {}\nnegtive samples: {}'.format(pos_samples_num, N-pos_samples_num))

    return labels, pos_samples_num, N-pos_samples_num


def flip_labels(labels, p, rng):
    if p <= 0:
        return

    flip_samples_indices = np.nonzero(rng.uniform(0, 1, size=labels.shape[0]) < p)[0]
    labels[flip_samples_indices] = - labels[flip_samples_indices]


def generate_instance(c_v, labels, dimension, N, pos_sample_num, neg_sample_num, rng, figname=""):
    '''
    generate random instances by normalizing the length of a
    vector of random-normal values (these distribute evenly on ball).
    :param c_v: concept of view v
    :param labels: labels of all samples before flipping
    :param dimension: dimension of instance space
    :param N: number of samples
    :param rng: random number generator
    :param figname: name of figure
    :return: samples in view v
    '''
    samples = np.empty(shape=(N, dimension))
    pos_sample_count = 0
    neg_sample_count = 0

    while pos_sample_count < pos_sample_num or neg_sample_count < neg_sample_num:
        random_vectors = rng.standard_normal(size=(N//2, dimension))
        c_v_pre = np.sign(np.matmul(random_vectors, c_v.T))
        if pos_sample_count >= pos_sample_num:
            ...
        else:
            pos_samples_indices = np.nonzero(c_v_pre > 0)[0]
            pos_end = min(pos_sample_num, pos_sample_count + pos_samples_indices.shape[0])
            samples[pos_sample_count:pos_end, :] = normalize(random_vectors[pos_samples_indices[0:pos_end - pos_sample_count], :], axis=1, norm='l2')
            pos_sample_count = pos_end

        if neg_sample_count >= neg_sample_num:
            ...
        else:
            start_neg_sample_index = pos_sample_num + neg_sample_count
            neg_samples_indices = np.nonzero(c_v_pre < 0)[0]
            neg_end = min(neg_sample_num + pos_sample_num, start_neg_sample_index + neg_samples_indices.shape[0])
            samples[start_neg_sample_index:neg_end, :] = normalize(random_vectors[neg_samples_indices[0:neg_end - start_neg_sample_index], :], axis=1, norm='l2')
            neg_sample_count = neg_end - pos_sample_num

    if figname == "./X_1_concept":
        plt.figure(1)
    else:
        plt.figure(2)

    plt.scatter(samples[0:pos_sample_num, 0], samples[0:pos_sample_num, 1], color='red')
    plt.scatter(samples[pos_sample_num:, 0], samples[pos_sample_num:, 1], color='blue')
    if abs(c_v[0]/c_v[1]) < 2:
        plt.plot([1.1, -1.1], [-1.1 * c_v[0] / c_v[1], 1.1 * c_v[0] / c_v[1]], color='black')
    else:
        plt.plot([-1.1 * c_v[1] / c_v[0], 1.1 * c_v[1] / c_v[0]], [1.1, -1.1], color='black')
    plt.arrow(0, 0, c_v[0]/2, c_v[1]/2, head_width=0.05, color='red')
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


def safe_div(x,y):
    if y:
        return x/y
    else:
        return 0


def cal_h_err(h_1, h_2, X_1, X_2, labels):
    '''

    :param h_1:
    :param h_2:
    :param X_1:
    :param X_2:
    :param labels:
    :return:
    '''
    h_1_pre = np.sign(np.matmul(X_1, h_1.T))
    h_2_pre = np.sign(np.matmul(X_2, h_2.T))
    disagreement_region = np.nonzero(h_1_pre - h_2_pre)[0]
    flip_ret = np.sign(np.random.uniform(-1, 1, disagreement_region.shape[0]))

    h_1_pre[disagreement_region] = flip_ret
    error = safe_div((np.nonzero(h_1_pre - labels)[0]).shape[0], labels.shape[0])
    return error


def algorithm_1(X_1, X_2, labels, H_1, H_2, beta, delta, rng):
    '''
    :param X_1: shape=(sample_N, dimension) instances in view 1
    :param X_2: shape=(sample_N, dimension) instances in view 2
    :param H_1: shape=(sample_N, dimension) hypotheses in view 1
    :param H_2: shape=(sample_N, dimension) hypotheses in view 2
    :param labels: labels of all samples
    :param beta:
    :param delta:
    :param unlabeled_samples_batch: batch size of unlabeled samples
    :return: hypothesis indices of h_1 and h_2
    '''

    '''
    k = H_s[x] where k = H_2.size * i + j
    represents the hypothesis (j,i)
    H_s : shape = (H_1.size * H_2.size,)
    '''
    H_s = np.full(H_1.shape[0] * H_2.shape[0], True)
    H_s = np.nonzero(H_s == True)[0]

    unlabeled_samples_num = np.log(H_s.size) / beta * np.log(1 / (beta * delta)) / 50
    unlabeled_samples_num = int(unlabeled_samples_num//unlabeled_samples_batch * unlabeled_samples_batch)

    '''
    statistics_record : shape (6, total_iterations)
    size of H_s
    average error of h in H_s
    Number of h in H_s where er(h)<0.5
    Average error of good h in H_s
    Number of h in H_s where er(h)>=0.5
    Average error of bad h in H_s
    '''
    statistics_record = np.zeros(shape=(6, unlabeled_samples_num//unlabeled_samples_batch - 1))

    for iteration, unlabeled_samples_count in enumerate(range(unlabeled_samples_init, unlabeled_samples_num, unlabeled_samples_batch)):
        print("\n-----------------\nThe {}-th iteration\n-----------------".format(iteration))
        print("Current number of unlabeled samples are {}.".format(unlabeled_samples_count))

        unlabeled_samples_indices = rng.integers(X_1.shape[0], size=unlabeled_samples_batch)

        for h_s_index, h_index in enumerate(H_s):
            h_1_index = h_index % H_2.shape[0]
            h_2_index = h_index // H_2.shape[0]
            h_1, h_2 = H_1[h_1_index, :], H_2[h_2_index, :]
            h_1_unlabeled_pre = np.sign(np.matmul(X_1[unlabeled_samples_indices, :], h_1.T))
            h_2_unlabeled_pre = np.sign(np.matmul(X_2[unlabeled_samples_indices, :], h_2.T))

            if not np.array_equal(h_1_unlabeled_pre, h_2_unlabeled_pre):
                H_s[h_s_index] = -1

        # update H_s and initialize error_list
        H_s = H_s[np.nonzero(H_s > 0)]
        error_list = np.zeros(H_s.shape[0])
        for h_s_index, h_index in enumerate(H_s):
            h_1_index = h_index % H_2.shape[0]
            h_2_index = h_index // H_2.shape[0]
            h_1, h_2 = H_1[h_1_index, :], H_2[h_2_index, :]

            error_list[h_s_index] = cal_h_err(h_1, h_2, X_1, X_2, labels)

        statistics_record[0, iteration] = H_s.shape[0]
        statistics_record[1, iteration] = np.average(error_list)
        statistics_record[2, iteration] = np.nonzero(error_list < 0.5)[0].size
        statistics_record[3, iteration] = safe_div(np.sum(error_list[error_list < 0.5]), statistics_record[2, iteration])
        statistics_record[4, iteration] = np.nonzero(error_list >= 0.5)[0].size
        statistics_record[5, iteration] = safe_div(np.sum(error_list[error_list >= 0.5]), statistics_record[4, iteration])

        print(f"The size of H_s is {statistics_record[0, iteration]: .0f}.")
        print(f"Average error of h in H_s is {statistics_record[1, iteration]}.")
        print(f"Number of h in H_s where er(h) < 0.5 is {statistics_record[2, iteration]: .0f}.")
        print(f"Average error of good h in H_s is {statistics_record[3, iteration]}.")
        print(f"Number of h in H_s where er(h) >= 0.5 is {statistics_record[4, iteration]: .0f}.")
        print(f"Average error of bad h in H_s is {statistics_record[5, iteration]}.")

    labeled_samples_indices = rng.integers(X_1.shape[0], size=budget)
    requested_labels = labels[labeled_samples_indices]
    labeled_x_1 = X_1[labeled_samples_indices, :]
    labeled_x_2 = X_2[labeled_samples_indices, :]
    error_list = np.zeros(shape=H_s.size)

    for h_s_index, h_index in enumerate(H_s):
        h_1_index = h_index % H_2.shape[0]
        h_2_index = h_index // H_2.shape[0]
        h_1, h_2 = H_1[h_1_index, :], H_2[h_2_index, :]
        h_1_pre = np.sign(np.matmul(labeled_x_1, h_1.T))
        h_2_pre = np.sign(np.matmul(labeled_x_2, h_2.T))
        disagreement_region = np.nonzero(h_1_pre - h_2_pre)[0]
        flip_ret = np.sign(np.random.uniform(-1, 1, disagreement_region.shape[0]))

        h_1_pre[disagreement_region] = flip_ret
        error_list[h_s_index] = ((np.nonzero(h_1_pre - requested_labels)[0]).shape[0]) / budget

    best_hypothesis_H_s_indices = np.nonzero(error_list == np.amin(error_list))[0]
    chosen_hypothesis_H_s_index = np.random.choice(best_hypothesis_H_s_indices)

    print(f"error_list: {error_list}")

    h_index = H_s[chosen_hypothesis_H_s_index]
    h_1_index = h_index % H_2.shape[0]
    h_2_index = h_index // H_2.shape[0]
    h_1, h_2 = H_1[h_1_index, :], H_2[h_2_index, :]
    error = cal_h_err(h_1, h_2, X_1, X_2, labels)

    print(f"find h_out with error er(h_out) = {error}.")
    plot_statistics(statistics_record)

    # return h_1, h_2


def plot_statistics(statistics_record):
    '''
    :param statistics_record: shape (6, total_iterations)
        size of H_s
        average error of h in H_s
        Number of h in H_s where er(h)<0.5
        Average error of good h in H_s
        Number of h in H_s where er(h)>=0.5
        Average error of bad h in H_s
    :param unlabeled_samples_batch
    '''

    plt.figure(3)
    plt.plot(unlabeled_samples_init + unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[0, :], color='blue')
    plt.xlabel('unlabeled samples')
    plt.ylabel('|H_s|')
    plt.savefig('H_s_size')

    plt.figure(4)
    plt.plot(unlabeled_samples_init + unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[1, :], color='blue')
    plt.xlabel('unlabeled samples')
    plt.ylabel('Average er(h) for h in H_s')
    plt.savefig('ave_err_h')

    plt.figure(5)
    plt.plot(unlabeled_samples_init + unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[2, :], color='blue'
             , label='good hypotheses')
    plt.plot(unlabeled_samples_init + unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[4, :], color='red'
             , label='bad hypotheses')
    plt.xlabel('unlabeled samples')
    plt.ylabel('number of good/bad h in H_s')
    plt.legend()
    plt.savefig('compare_num_h')

    plt.figure(6)
    plt.plot(unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[3, :], color='blue'
             , label='good hypotheses')
    plt.plot(unlabeled_samples_batch * np.arange(statistics_record.shape[1]), statistics_record[5, :], color='red'
             , label='bad hypotheses')
    plt.xlabel('unlabeled samples')
    plt.ylabel('Average er(h) for good/bad h in H_s')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('compare_err_h')


def main():
    H_1 = generate_H_v(H_1_delta, dimension_1)
    H_2 = generate_H_v(H_2_delta, dimension_2)
    # H_indices = generate_hypotheses(H_1.shape[0], H_2.shape[0])
    print(f'Total number of hypotheses: {H_1.shape[0] * H_2.shape[0]}')

    if random_seed > 0:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()

    c_1_index, c_2_index = choose_concept(H_1.shape[0], H_2.shape[0], rng)
    labels, pos_sample_num, neg_sample_num = generate_labels(sample_N, pos_p, rng)
    X_1 = generate_instance(H_1[c_1_index, :], labels, dimension_1, sample_N, pos_sample_num, neg_sample_num, rng, "./X_1_concept")
    X_2 = generate_instance(H_2[c_2_index, :], labels, dimension_2, sample_N, pos_sample_num, neg_sample_num, rng, "./X_2_concept")
    # beta = min(alpha*eps/2, eps)
    flip_labels(labels, flip_p, rng)
    beta = eps
    algorithm_1(X_1, X_2, labels, H_1, H_2, beta, 1 - confidence, rng)
    # test_accuracy(h_1, h_2, X_1, X_2, labels)

if __name__ == '__main__':
    main()
