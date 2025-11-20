import numpy as np
from scipy.linalg import expm, logm

deg2rad = np.pi / 180
rad2deg = 1 / deg2rad

def hat(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def inv_hat(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


# (a, n): a = angle, n = vecteur unitaire (axe de rotation)

def quat2axang(q):
    q = q / sum(q*q)
    mu = 2 * np.arccos(min(1, q[0]))
    if (q[0] >= 1) :
        n = np.zeros(3)
    else :
        n = q[1:4] / np.sqrt(1 - q[0]*q[0])
    return (mu, n)

def dcm2axang(R):
    t = np.trace(R)
    mu = np.arccos((t - 1) / 2)
    n = 0.5 * inv_hat(R.T - R) / np.sin(mu)
    return (mu, n)


def axangrot(ax, u):            # left hand rotation du vecteur u d'un angle mu selon l'axe n
    mu, n = ax
    v = (1 - np.cos(mu)) * n * np.dot(n, u) + u * np.cos(mu) - np.cross(n, u) * np.sin(mu)
    return v


# Quaternions

def axang2quat(ax):
    mu, n = ax
    return np.concatenate([[np.cos(mu/2)], n * np.sin(mu/2)])
    
def eul2quat(a):
    qx = np.array([np.cos(a[0]/2), np.sin(a[0]/2), 0, 0])       # roll angle : phi
    qy = np.array([np.cos(a[1]/2), 0, np.sin(a[1]/2), 0])       # pitch angle : theta
    qz = np.array([np.cos(a[2]/2), 0, 0, np.sin(a[2]/2)])       # yaw angle : psi
    return quatmul(qz, quatmul(qy, qx))

def vect2quat(u):
    return np.concatenate([[0], u])

def quatmul(p, q):
    m = np.array([[p[0], -p[1], -p[2], -p[3]],
                  [p[1],  p[0], -p[3],  p[2]],
                  [p[2],  p[3],  p[0], -p[1]], 
                  [p[3], -p[2],  p[1],  p[0]]])
    return m@q

def quatinv(q):
    norm = np.sum(q*q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        return np.concatenate([[q[0]], -q[1:4]]) / norm

def quatrot(q, u):
    v = quatmul(quatmul(quatinv(q), vect2quat(u)), q)
    return v[1:]

def quatdist(q1, q2):
    return 2 * np.arccos(min(1,abs(sum(q1*q2))))                 # donne une valeur d'angle entre 0 et PI

def quatdist2(q1, q2):
    return 1 - abs(sum(q1*q2))                                   # donne une valeur d'angle entre 0 et 1

def quaterr(q1, q2):
    mu, n = quat2axang(quatmul(quatinv(q2), q1))
    return mu*n


# Matrices de Rotation (DCM)

def axang2dcm(ax):
    mu, n = ax
    dcm = (1 - np.cos(mu)) * np.outer(n, n) + np.cos(mu) * np.identity(3) - np.sin(mu) * hat(n)
    return dcm

def quat2dcm(q):
    q = q / quatnorm(q)
    q0 = q[0]
    w = q[1:4]
    dcm = (2 * np.outer(w, w) + (q0**2 - w @ w) * np.identity(3) - 2 * q0 * hat(w)) 
    return dcm

def eul2dcm(a):
    Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), np.sin(a[0])], [0, -np.sin(a[0]), np.cos(a[0])]])
    Ry = np.array([[np.cos(a[1]), 0, -np.sin(a[1])], [0, 1, 0], [np.sin(a[1]), 0, np.cos(a[1])]])
    Rz = np.array([[np.cos(a[2]), np.sin(a[2]), 0], [-np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])               
    return Rx @ Ry @ Rz


def dcmerr(R1, R2):
    return inv_hat(logm(R2@R1.T))

def dcmdist(R1, R2):
    return np.linalg.norm(dcmerr(R1, R2))               # donne une valeur d'angle entre 0 et PI


def dcmerr2(R, Rd):
    return 0.5 * inv_hat(Rd@R.T - R@Rd.T)

def dcmdist2(R, Rd):
    return np.linalg.norm(np.identity(3) - Rd@R.T)     # donne une valeur ente 0 et 2V2

def dcmdist4(R, Rd):
    return  0.5 * np.trace(np.identity(3) - Rd@R.T)


def dcmdist3(R, Rd):
    dist = 2 - np.sqrt(1 + np.trace(Rd@R.T))
    return dist

def dcmerr3(R, Rd):
    err = inv_hat(Rd@R.T - R@Rd.T) / np.sqrt(1 + np.trace(Rd@R.T))
    return err


# Angles d'Euler (phi, theta, psi) selon axes (x, y, z) : z en premier

def quat2eul(q):
    phi = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
    theta = -np.arcsin(min(1, 2 * (q[1] * q[3] - q[0] * q[2])))
    psi = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)
    return np.array([phi, theta, psi])
    
def dcm2eul(R):
    phi = np.arctan2(R[1, 2], R[2, 2])
    theta = -np.arcsin(R[0, 2])
    psi = np.arctan2(R[0, 1], R[0, 0])
    return np.array([phi, theta, psi])


def euldist(e1, e2):                                    # donne une valeur d'angle entre 0 et PI
    diff = e1 - e2
    d = np.fmin(np.abs(diff), 2*np.pi - np.abs(diff))
    r = np.linalg.norm(d) / np.sqrt(3)
    return r
