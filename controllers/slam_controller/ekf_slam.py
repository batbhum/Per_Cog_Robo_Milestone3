"""
ekf_slam.py - Extended Kalman Filter SLAM.

State: [x, y, theta, lx_1, ly_1, ..., lx_n, ly_n]^T

PREDICT:
    x' = x + v*cos(θ)*dt
    y' = y + v*sin(θ)*dt
    θ' = θ + ω*dt
    P' = F*P*F^T + Q

UPDATE (per landmark observation (r, bearing)):
    Expected: r_hat, b_hat from current state
    H = Jacobian of observation model
    S = H*P*H^T + R
    K = P*H^T*S^-1
    mu += K*(z - z_hat)
    P = (I - K*H)*P

DATA ASSOCIATION: Mahalanobis distance to nearest existing landmark.
"""
import math
from utils import normalize_angle


class EKFSLAM:
    def __init__(self, init_x, init_y, init_theta):
        self.mu = [init_x, init_y, init_theta]
        self.P = [[0.001,0,0],[0,0.001,0],[0,0,0.001]]
        self.n_landmarks = 0
        self.sigma_v = 0.05
        self.sigma_omega = 0.02
        self.sigma_r = 0.1
        self.sigma_phi = 0.05
        self.association_threshold = 2.0

    def _n(self):
        return 3 + 2 * self.n_landmarks

    def predict(self, v, omega, dt):
        theta = self.mu[2]
        n = self._n()
        self.mu[0] += v * math.cos(theta) * dt
        self.mu[1] += v * math.sin(theta) * dt
        self.mu[2] = normalize_angle(self.mu[2] + omega * dt)

        f02 = -v * math.sin(theta) * dt
        f12 =  v * math.cos(theta) * dt

        # F*P
        new_P = [row[:] for row in self.P]
        for j in range(n):
            new_P[0][j] = self.P[0][j] + f02 * self.P[2][j]
            new_P[1][j] = self.P[1][j] + f12 * self.P[2][j]
        # (F*P)*F^T
        res = [row[:] for row in new_P]
        for i in range(n):
            res[i][0] = new_P[i][0] + f02 * new_P[i][2]
            res[i][1] = new_P[i][1] + f12 * new_P[i][2]
        # Add Q
        res[0][0] += (self.sigma_v * dt) ** 2
        res[1][1] += (self.sigma_v * dt) ** 2
        res[2][2] += (self.sigma_omega * dt) ** 2
        self.P = res

    def update(self, observations):
        for obs_r, obs_b in observations:
            obs_b = normalize_angle(obs_b)
            best_idx, best_m, best_inn, best_H, best_S = -1, float('inf'), None, None, None
            for j in range(self.n_landmarks):
                inn, H, S = self._innovation(j, obs_r, obs_b)
                if inn is None: continue
                m = self._mahal(inn, S)
                if m < best_m:
                    best_m, best_idx, best_inn, best_H, best_S = m, j, inn, H, S
            if best_m < self.association_threshold and best_idx >= 0:
                self._apply_update(best_inn, best_H, best_S)
            else:
                self._add_landmark(obs_r, obs_b)

    def _innovation(self, j, obs_r, obs_b):
        n = self._n()
        lx = self.mu[3+2*j]; ly = self.mu[3+2*j+1]
        rx, ry, rt = self.mu[0], self.mu[1], self.mu[2]
        dx = lx-rx; dy = ly-ry; q = dx*dx+dy*dy
        if q < 1e-10: return None, None, None
        r_e = math.sqrt(q)
        b_e = normalize_angle(math.atan2(dy,dx) - rt)
        inn = [obs_r - r_e, normalize_angle(obs_b - b_e)]
        H = [[0.0]*n for _ in range(2)]
        H[0][0]=-dx/r_e; H[0][1]=-dy/r_e
        H[1][0]=dy/q;    H[1][1]=-dx/q; H[1][2]=-1.0
        li = 3+2*j
        H[0][li]=dx/r_e; H[0][li+1]=dy/r_e
        H[1][li]=-dy/q;  H[1][li+1]=dx/q
        R = [[self.sigma_r**2,0],[0,self.sigma_phi**2]]
        S = self._S(H, R, n)
        return inn, H, S

    def _S(self, H, R, n):
        HP = [[sum(H[i][k]*self.P[k][j] for k in range(n)) for j in range(n)] for i in range(2)]
        S = [[sum(HP[i][k]*H[j][k] for k in range(n))+R[i][j] for j in range(2)] for i in range(2)]
        return S

    def _mahal(self, inn, S):
        det = S[0][0]*S[1][1]-S[0][1]*S[1][0]
        if abs(det)<1e-10: return float('inf')
        Si = [[S[1][1]/det,-S[0][1]/det],[-S[1][0]/det,S[0][0]/det]]
        t0 = Si[0][0]*inn[0]+Si[0][1]*inn[1]
        t1 = Si[1][0]*inn[0]+Si[1][1]*inn[1]
        return math.sqrt(max(0, inn[0]*t0+inn[1]*t1))

    def _apply_update(self, inn, H, S):
        n = self._n()
        det = S[0][0]*S[1][1]-S[0][1]*S[1][0]
        if abs(det)<1e-10: return
        Si = [[S[1][1]/det,-S[0][1]/det],[-S[1][0]/det,S[0][0]/det]]
        PHt = [[sum(self.P[i][k]*H[j][k] for k in range(n)) for j in range(2)] for i in range(n)]
        K = [[PHt[i][0]*Si[0][j]+PHt[i][1]*Si[1][j] for j in range(2)] for i in range(n)]
        for i in range(n):
            self.mu[i] += K[i][0]*inn[0]+K[i][1]*inn[1]
        self.mu[2] = normalize_angle(self.mu[2])
        KH = [[K[i][0]*H[0][j]+K[i][1]*H[1][j] for j in range(n)] for i in range(n)]
        new_P = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = sum(((-KH[i][k]+1.0 if i==k else -KH[i][k]))*self.P[k][j] for k in range(n))
                new_P[i][j] = s
        self.P = new_P

    def _add_landmark(self, obs_r, obs_b):
        rx,ry,rt = self.mu[0],self.mu[1],self.mu[2]
        self.mu += [rx+obs_r*math.cos(rt+obs_b), ry+obs_r*math.sin(rt+obs_b)]
        n_old = self._n(); self.n_landmarks += 1; n_new = self._n()
        new_P = [[0.0]*n_new for _ in range(n_new)]
        for i in range(n_old):
            for j in range(n_old):
                new_P[i][j] = self.P[i][j]
        new_P[n_new-2][n_new-2] = 1.0
        new_P[n_new-1][n_new-1] = 1.0
        self.P = new_P

    def get_robot_pose(self):
        return self.mu[0], self.mu[1], self.mu[2]

    def get_robot_covariance(self):
        return [[self.P[i][j] for j in range(3)] for i in range(3)]

    def get_landmarks(self):
        return [(self.mu[3+2*i], self.mu[3+2*i+1]) for i in range(self.n_landmarks)]
