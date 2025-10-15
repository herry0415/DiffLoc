import numpy as np
import scipy
import numpy.matlib as ml
import transforms3d.quaternions as txq
from math import sin, cos, atan2, sqrt


def filter_overflow_ts(filename, ts_raw):
    #
    # file_data = pd.read_csv(filename)
    file_data = np.loadtxt(filename)

    pose_timestamps = file_data[:, 0]
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))

    return ts_filted


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = ml.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        format: degree or radians
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = np.dot(R_z, np.dot(R_y, R_x))
    return R_zyx


def so3_to_euler(poses_in):
    N = len(poses_in)
    poses_out = np.zeros((N, 4, 4))
    for i in range(N):
        poses_out[i, :, :] = build_se3_transform([poses_in[i, 0], poses_in[i, 1], poses_in[i, 2],
                                                  poses_in[i, 3], poses_in[i, 4], poses_in[i, 5]])

    return poses_out


def filter_overflow_xmu(filename, ts_raw):  # 滤波函数
    file_data = np.loadtxt(filename)
    pose_timestamps = file_data[:, 0]
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))

    return ts_filted


def interpolate_pose_xmu(gt_filename, ts_raw):  # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename)
    ground_truth = ground_truth[np.logical_not(np.any(np.isnan(ground_truth), 1))]
    interp = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(ts_raw)

    return pose_gt


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))

    return q


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def process_poses(poses_in):
    poses_out = np.zeros((len(poses_in), 7))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    align_R = np.eye(3)

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        poses_out[i, 3:] = q

    return poses_out


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted = pred_p
        groundtruth = gt_p
    else:
        predicted = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted = pred_q
        groundtruth = gt_q
    else:
        predicted = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))
    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1
    # error = 2 * np.arccos(d) * 180 / np.pi0
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

    d = np.abs(np.dot(groundtruth, predicted))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error