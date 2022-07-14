import matplotlib.colors
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
parser.add_argument('--flip_p', type=float, nargs='?', default=0,
                    help='probability of flipping the label after generating a sample')
parser.add_argument('--alpha', type=float, nargs='?', default=2,
                    help='value of alpha of alpha-expansion')
# parser.add_argument('--alpha', type=float, nargs='?', default=2,
#                     help='value of alpha of alpha-expansion')
# parser.add_argument('--alpha', type=float, nargs='?', default=2,
#                     help='value of alpha of alpha-expansion')
# parser.add_argument('--alpha', type=float, nargs='?', default=2,
#                     help='value of alpha of alpha-expansion')
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
flip_p = opt.flip_p
random_seed = opt.random_seed
alpha = opt.alpha
round_point_approx = 0.0000001
c = 1
c_0 = 1/0.35
lamb = 1
k = 1 / lamb + 1
v = max(dimension_1 + 1, dimension_2 + 1)


def cartesian_product_transpose(arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)
    out = np.empty(rows * cols, dtype=dtype)
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

    H_v = np.ones(shape=(H_v_delta ** (dimension - 1), dimension))
    H_v_polar = np.zeros(shape=(H_v_delta, dimension - 1))

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

    return labels, pos_samples_num, N - pos_samples_num


def flip_labels(labels, p, rng):
    if p <= 0:
        return

    flip_samples_indices = np.nonzero(rng.uniform(0, 1, size=labels.shape[0]) < p)[0]
    labels[flip_samples_indices] = - labels[flip_samples_indices]


def generate_instance(c_v, dimension, N, pos_sample_num, neg_sample_num, rng, labels, figname="", **kwargs):
    '''
    default: uniform distribution on unit sphere
        'cube': uniform distribution in unit cube
        'rect': uniform distribution in rectangular
        'ball': uniform distribution in ball
    :param c_v: concept of view v
    :param dimension: dimension of instance space
    :param N: number of samples
    :param pos_sample_num: num of positive samples
    :param neg_sample_num: num of negative samples
    :param rng: random number generator
    :param figname: name of figure
    :return: samples in view v
    '''
    samples = np.empty(shape=(N, dimension))
    pos_sample_count = 0
    neg_sample_count = 0

    while pos_sample_count < pos_sample_num or neg_sample_count < neg_sample_num:
        if kwargs["distribution"] == "cube":
            random_vectors = rng.uniform(-1, 1, size=(N // 2, dimension))
        elif kwargs["distribution"] == "ball":
            random_vectors = rng.standard_normal(size=(N // 2, dimension))
            random_vectors = normalize(random_vectors, axis=1, norm='l2')
            radius = np.power(rng.uniform(0, 1, N//2), 1/dimension)
            random_vectors *= np.expand_dims(radius, 0).T
        else:
            random_vectors = rng.standard_normal(size=(N // 2, dimension))
            random_vectors = normalize(random_vectors, axis=1, norm='l2')

        c_v_pre = np.sign(np.matmul(random_vectors, c_v.T))

        if pos_sample_count >= pos_sample_num:
            ...
        else:
            pos_samples_indices = np.nonzero(c_v_pre > 0)[0]
            pos_end = min(pos_sample_num, pos_sample_count + pos_samples_indices.shape[0])

            samples[pos_sample_count:pos_end, :] = random_vectors[pos_samples_indices[0:pos_end - pos_sample_count], :]
            pos_sample_count = pos_end

        if neg_sample_count >= neg_sample_num:
            ...
        else:
            start_neg_sample_index = pos_sample_num + neg_sample_count
            neg_samples_indices = np.nonzero(c_v_pre < 0)[0]
            neg_end = min(neg_sample_num + pos_sample_num, start_neg_sample_index + neg_samples_indices.shape[0])
            samples[start_neg_sample_index:neg_end, :] = random_vectors[neg_samples_indices[0:neg_end - start_neg_sample_index], :]
            neg_sample_count = neg_end - pos_sample_num

    if figname == "./X_1_concept":
        plt.figure(1)
    else:
        plt.figure(2)

    cmp = matplotlib.colors.ListedColormap(['b','r'])
    norm = matplotlib.colors.BoundaryNorm([-1, 0, 1], cmp.N)
    # plt.scatter(samples[0:pos_sample_num, 0], samples[0:pos_sample_num, 1], color='red')
    # plt.scatter(samples[pos_sample_num:, 0], samples[pos_sample_num:, 1], color='blue')
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap=cmp, norm=norm)
    if abs(c_v[0]/c_v[1]) < 2:
        plt.plot([1.1, -1.1], [-1.1 * c_v[0] / c_v[1], 1.1 * c_v[0] / c_v[1]], color='black')
    else:
        plt.plot([-1.1 * c_v[1] / c_v[0], 1.1 * c_v[1] / c_v[0]], [1.1, -1.1], color='black')
    plt.arrow(0, 0, c_v[0]/2, c_v[1]/2, head_width=0.05, color='red')
    plt.savefig(figname)

    # plt.show()

    return samples


def composite_h(h_1, h_2, x_1, x_2):
    if np.sign(np.vdot(h_1, x_1)) == np.sign(np.vdot(h_2, x_2)):
        return np.sign(np.vdot(h_1, x_1))
    else:
        if np.random.uniform(0, 1) > 0.5:
            return 1
        else:
            return -1


def safe_div(x, y):
    if y:
        return x/y
    else:
        print('Divided by 0. Return 0 as the result.')
        return 0


def cal_h_err(h, X, labels):
    h_pre = np.sign(np.matmul(X, h.T))
    error = safe_div((np.nonzero(h_pre - labels)[0]).shape[0], labels.shape[0])
    return error


def cal_h_pos_err(h_1, h_2, X_1, X_2, labels):
    h_1_pre = np.sign(np.matmul(X_1, h_1.T))
    h_2_pre = np.sign(np.matmul(X_2, h_2.T))
    ret = np.zeros(labels.shape)
    error_count = 0
    for index, label in enumerate(labels):
        if h_1_pre[index] == h_2_pre[index] and h_1_pre[index] == 1:
            if label == 1:
                ...
            else:
                error_count += 1
        else:
            if label == 1:
                error_count += 1
            else:
                ...
    return error_count/labels.shape[0]


def cal_h_neg_err(h_1, h_2, X_1, X_2, labels):
    h_1_pre = np.sign(np.matmul(X_1, h_1.T))
    h_2_pre = np.sign(np.matmul(X_2, h_2.T))
    ret = np.zeros(labels.shape)
    error_count = 0
    for index, label in enumerate(labels):
        if h_1_pre[index] == h_2_pre[index] and h_1_pre[index] == -1:
            if label == 1:
                error_count += 1
            else:
                ...
        else:
            if label == 1:
                ...
            else:
                error_count += 1
    return error_count / labels.shape[0]


def train(instances: np.ndarray, labels, H):
    best_h, best_err = H[0, :], 1
    for index in range(H.shape[0]):
        h_err = cal_h_err(H[index, :], instances, labels)
        if h_err < best_err:
            best_h, best_err = H[index, :], h_err

    return best_h, best_err


def algorithm(X_1, X_2, labels, H_1, H_2, rng):
    '''
    :param X_1: shape=(sample_N, dimension) instances in view 1
    :param X_2: shape=(sample_N, dimension) instances in view 2
    :param labels: labels of all samples
    :param H_1: shape=(sample_N, dimension) hypotheses in view 1
    :param H_2: shape=(sample_N, dimension) hypotheses in view 2
    :param rng: random number generator
    :return: hypothesis indices of h_1 and h_2
    '''
    c_1 = 2 * np.power(c_0, -1 / lamb) * lamb * np.power(lamb + 1, -1 - 1 / lamb)
    c_2 = (5 * alpha + 8) / (6 * alpha + 8)
    # s = int(np.ceil(2*np.log(1/(8 * eps)) / (np.log(1/c_2) ) ) )
    s = 5
    # m_i = int(np.power(256, k) * c / (c_1 * c_1) * ( v + np.log(16 * (s + 1) / (1 - confidence) ) ) )
    m_i = 10

    print(f"c={c}, c_0={c_0}, c_1={c_1}, c_2={c_2}\nalpha={alpha}, lambda={lamb}, k={k},\nV={v}, s={s}, m_i={m_i}")

    # Initialize U and L
    unlabeled_samples_mask = np.ones(shape=(X_1.shape[0]), dtype=bool)
    labeled_samples_mask = np.zeros(shape=(X_1.shape[0]), dtype=bool)
    unlabeled_samples_size = np.nonzero(unlabeled_samples_mask)[0].size
    print("Current number of unlabeled samples are {}.".format(unlabeled_samples_size))

    # query the labels of m_0 instances drawn randomly from U
    queried_instances_indices = (np.nonzero(unlabeled_samples_mask)[0])[rng.integers(0, unlabeled_samples_size, size=m_i)]
    print(f"# of drawn instances: {queried_instances_indices.shape[0]}")

    # Update labeled data set L
    labeled_samples_mask[queried_instances_indices] = True
    print(f"Current # of labeled samples: {np.nonzero(labeled_samples_mask)[0].size}")

    # Train the classifier h_v^0 by minimizing the empirical risk with L in each view
    cur_best_h_1, h_1_err = train(X_1[labeled_samples_mask], labels[labeled_samples_mask], H_1)
    cur_best_h_2, h_2_err = train(X_2[labeled_samples_mask], labels[labeled_samples_mask], H_2)

    statistics_record = np.zeros(shape=(3, s+1))
    statistics_record[0, 0] = np.nonzero(labeled_samples_mask)[0].size
    statistics_record[1, 0] = cal_h_pos_err(cur_best_h_1, cur_best_h_2, X_1, X_2, labels)
    statistics_record[2, 0] = cal_h_neg_err(cur_best_h_1, cur_best_h_2, X_1, X_2, labels)

    for iteration in range(1, s+1):

        # apply h_1^{i-1} and h_2^{i-1} to U and find out contention point set Q_i
        h_1_pre = np.sign(np.matmul(X_1[unlabeled_samples_mask], cur_best_h_1.T))
        h_2_pre = np.sign(np.matmul(X_2[unlabeled_samples_mask], cur_best_h_2.T))
        if np.nonzero(h_1_pre - h_2_pre)[0].size == 0:
            print("No contention points")
            queried_points_indices = np.array([])
        else:
            queried_points_indices = rng.choice(np.nonzero(h_1_pre - h_2_pre)[0], size=m_i)
        print(f"\nIteration: {iteration}\n   Contention Points: {np.nonzero(h_1_pre - h_2_pre)[0].size}")

        # Query m_i labels drawn randomly from contention points
        # Add them into L and delete from U
        labeled_samples_mask[queried_points_indices] = True
        unlabeled_samples_mask[queried_points_indices] = False

        # Obtain U-Q_i
        U_copy = np.copy(unlabeled_samples_mask)
        if np.nonzero(h_1_pre - h_2_pre)[0].size != 0:
            U_copy[np.nonzero(h_1_pre - h_2_pre)[0]] = False

        # Query (2^i - 1)m_i labels from U-Q_i, add them into L and delete them from U
        queried_points_indices = rng.choice(np.nonzero(U_copy)[0], size=(np.power(2, iteration) - 1) * m_i)
        labeled_samples_mask[queried_points_indices] = True
        unlabeled_samples_mask[queried_points_indices] = False
        print(f"|L| = {np.nonzero(labeled_samples_mask)[0].shape[0]}")
        print(f"|U| = {np.nonzero(unlabeled_samples_mask)[0].shape[0]}")

        # Train the classifier h_v^i by minimizing tghe empirical rish with L in each view
        cur_best_h_1, h_1_err = train(X_1[labeled_samples_mask], labels[labeled_samples_mask], H_1)
        cur_best_h_2, h_2_err = train(X_2[labeled_samples_mask], labels[labeled_samples_mask], H_2)

        statistics_record[0, iteration] = np.nonzero(labeled_samples_mask)[0].shape[0]
        statistics_record[1, iteration] = cal_h_pos_err(cur_best_h_1, cur_best_h_2, X_1, X_2, labels)
        statistics_record[2, iteration] = cal_h_neg_err(cur_best_h_1, cur_best_h_2, X_1, X_2, labels)

    plt.figure(10)
    plt.plot(statistics_record[0, :], statistics_record[1, :])
    plt.xlabel('labeled samples')
    plt.ylabel('average error of h_pos')
    plt.savefig("h_pos_err_vs_label_cost.png")

    plt.figure(11)
    plt.plot(statistics_record[0, :], statistics_record[2, :])
    plt.xlabel('labeled samples')
    plt.ylabel('average error of h_neg')
    plt.savefig("h_neg_err_vs_label_cost.png")


def main():
    H_1 = generate_H_v(H_1_delta, dimension_1)
    H_2 = generate_H_v(H_2_delta, dimension_2)
    # H_indices = generate_hypotheses(H_1.shape[0], H_2.shape[0])
    # print(f'Total number of hypotheses: {H_1.shape[0] * H_2.shape[0]}')

    if random_seed > 0:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()

    c_1_index, c_2_index = choose_concept(H_1.shape[0], H_2.shape[0], rng)
    labels, pos_sample_num, neg_sample_num = generate_labels(sample_N, pos_p, rng)
    flip_labels(labels, flip_p, rng)
    X_1 = generate_instance(H_1[c_1_index, :], dimension_1, sample_N, pos_sample_num, neg_sample_num, rng, labels, "./X_1_concept_zhou", distribution="sphere")
    X_2 = generate_instance(H_2[c_2_index, :], dimension_2, sample_N, pos_sample_num, neg_sample_num, rng, labels, "./X_2_concept_zhou", distribution="sphere")

    algorithm(X_1, X_2, labels, H_1, H_2, rng)
    # statistics_record = algorithm_1(X_1, X_2, labels, H_1, H_2, beta, 1 - confidence, rng)
    # plot_statistics(statistics_record)


if __name__ == '__main__':
    main()
