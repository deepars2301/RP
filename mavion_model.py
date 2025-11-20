import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from rotation import hat, eul2quat, vect2quat, quatinv, quatmul, quatrot
import yaml

class Mavion:
    
    def __init__(self, file="mavion.yaml"):
        stream = open(file, "r")
        drone_data = yaml.load(stream, Loader=yaml.Loader)

        self.NAME = drone_data['NAME']
        self.G = drone_data['G']
        self.RHO = drone_data['RHO']
        self.MASS = drone_data['MASS']
        self.CHORD = drone_data['CHORD']
        self.WINGSPAN = drone_data['WINGSPAN']
        self.WET_SURFACE = drone_data['WET_SURFACE']
        self.DRY_SURFACE = drone_data['DRY_SURFACE']
        self.PROP_RADIUS = drone_data['PROP_RADIUS']
        self.INERTIA = np.diag(drone_data["INERTIA"])

        self.PROP_KP = drone_data['PROP_KP']
        self.PROP_KM = drone_data['PROP_KM']
        self.INERTIA_PROP_X = drone_data['INERTIA_PROP_X']
        self.INERTIA_PROP_N = drone_data['INERTIA_PROP_N']

        self.ELEVON_MEFFICIENCY = np.array(drone_data['ELEVON_MEFFICIENCY'])
        self.ELEVON_FEFFICIENCY = np.array(drone_data['ELEVON_FEFFICIENCY'])
        self.MAX_ROTOR_SPD = drone_data['MAX_ROTOR_SPD']
        self.MAX_FLAP_DEF = drone_data['MAX_FLAP_DEF']

        # geometric parameters
        self.P_P1_CG = np.array(drone_data["P_P1_CG"]) * np.array([self.CHORD, self.WINGSPAN, 0])
        self.P_P2_CG = np.array(drone_data["P_P2_CG"]) * np.array([self.CHORD, self.WINGSPAN, 0])
        self.P_A1_CG = np.array(drone_data["P_A1_CG"]) * np.array([self.CHORD, self.WINGSPAN, 0])
        self.P_A2_CG = np.array(drone_data["P_A2_CG"]) * np.array([self.CHORD, self.WINGSPAN, 0])

        self.PHI_n = drone_data["PHI_n"]
        self.Cd0 = 0.1
        self.Cy0 = 0.1
        self.dR = -0.1 * self.CHORD
        self.PHI_fv = np.diag([self.Cd0, self.Cy0, (2 * np.pi + self.Cd0)])
        self.PHI_mv = np.array([[0, 0, 0], 
            [0, 0, -self.dR * (2 * np.pi + self.Cd0) / self.CHORD], 
            [0, self.dR * self.Cy0 / self.WINGSPAN, 0]])
        self.PHI_fw = self.PHI_mv.T
        self.PHI_mw = 1/2 * np.diag([0.5, 0.5, 0.5])


    def thrust(self, dx):
        kp = self.PROP_KP
        km = self.PROP_KM
        # prop forces computation / notice that negative thrust is not implemented
        T = kp * dx**2 * np.array([1, 0, 0])
        # prop moments computation
        N = np.sign(dx) * km * dx**2 * np.array([1, 0, 0])
        return np.array([T, N])

    def aero(self, vinf, rot, T, de):
        # data extraction from self struct
        PHI_fv = self.PHI_fv
        PHI_fw = self.PHI_fw
        PHI_mv = self.PHI_mv
        PHI_mw = self.PHI_mw
        PHI_n  = self.PHI_n
        RHO    = self.RHO
        Swet   = self.WET_SURFACE
        Sdry   = self.DRY_SURFACE
        chord  = self.CHORD
        ws     = self.WINGSPAN
        Prop_R = self.PROP_RADIUS
        Thetam = self.ELEVON_MEFFICIENCY
        Thetaf = self.ELEVON_FEFFICIENCY

        # derivative data
        Sp = np.pi * Prop_R**2
        # computation of total wing section area
        S = Swet + Sdry
        # computation of chord matrix
        B = np.array([ws, chord, ws])

        # eta computation
        eta = np.sqrt(vinf@vinf + PHI_n * (B*rot)@(B*rot))
        qv = 0.5*RHO*S*eta

        # Force computation
        # airfoil contribution
        Fb_a = -qv * (PHI_fv@vinf + PHI_fw@(B*rot)) - 1/2 * Swet/Sp * PHI_fv@T
        # elevon contribution
        Fb_e = qv * (PHI_fv@np.cross(Thetaf*de, vinf) + PHI_fw@(np.cross(Thetaf*de, B*rot)))  + 1/2 * Swet/Sp * PHI_fv@np.cross(Thetaf*de, T)
        Fb = Fb_a + Fb_e

        # Moment computation
        # airfoil contribution
        Mb_a = -qv * B*(PHI_mv@vinf + PHI_mw@(B*rot)) - 1/2 * Swet/Sp * B*(PHI_mv@T)
        # elevon contribution
        Mb_e = qv * B*(PHI_mv@np.cross(Thetam*de, vinf) + PHI_mw@(np.cross(Thetam*de, B*rot))) + 1/2 * Swet/Sp * B*(PHI_mv@np.cross(Thetam*de, T))
        Mb = Mb_a + Mb_e

        return np.array([Fb, Mb])


    def dyn(self, x, u, w):

        G = self.G
        MASS = self.MASS
        INERTIA = self.INERTIA
        # position of propellers wrt center of gravity
        P_P1_CG = self.P_P1_CG
        P_P2_CG = self.P_P2_CG
        # position of aerodynamic wrenches wrt center of gravity
        P_A1_CG = self.P_A1_CG
        P_A2_CG = self.P_A2_CG
        # propeller blade inertia
        INERTIA_PROP_X = self.INERTIA_PROP_X
        INERTIA_PROP_N = self.INERTIA_PROP_N

        # state demultiplexing
        quat = x[3:7]        # quaternion de passage de ned (ref terrestre) à frd (ref avion) = q_frd/ned   
        vel = x[7:10]        # vitesse par rapport au sol exprimée dans le repere terre (ned)
        rot = x[10:13]       # vitesse de rotation dans le repere avion (body / frd)
        
        [T1, N1] = self.thrust(-u[0])       
        [T2, N2] = self.thrust(u[1])
        tau1 = N1 - (rot[0] - u[0]) * (INERTIA_PROP_X - INERTIA_PROP_N) * np.array([0, rot[2], -rot[1]])
        tau2 = N2 - (rot[0] + u[1]) * (INERTIA_PROP_X - INERTIA_PROP_N) * np.array([0, rot[2], -rot[1]])

        vrel = quatrot(quat, vel - w)       # vitesse aéro exprimée dans le repere avion (frd)
        [F1, M1] = self.aero(vrel, rot, T1, u[2])
        [F2, M2] = self.aero(vrel, rot, T2, u[3])

        Fb = T1 + F1 + T2 + F2
        Mb = M1 + tau1 + np.cross(P_A1_CG, F1) + np.cross(P_P1_CG, T1) + M2 + tau2 + np.cross(P_A2_CG, F2) + np.cross(P_P2_CG, T2) 

        dpdt = vel
        dqdt = 0.5 * quatmul(quat, vect2quat(rot))
        dvdt = quatrot(quatinv(quat), Fb) / MASS + np.array([0, 0, G]) # exprimée dans le repere terrestre (ned)
        drdt = np.linalg.inv(INERTIA) @ (Mb - hat(rot)@(INERTIA@rot))

        return np.concatenate([dpdt, dqdt, dvdt, drdt])

    def trim(self, target):
        vh, vz = target
        x = np.zeros(13)
        x[7] = vh
        x[9] = vz

        def func(y):
            dx, de, theta = y
            x[3:7] = eul2quat([0, theta, 0])
            u = np.array([dx, dx, de, de])
            dq_dt = self.dyn(x, u, np.zeros(3))
            return dq_dt[7], dq_dt[9], dq_dt[11] # (v_north, vz, q)

        y0 = np.array([100, 0, 0])
        y = fsolve(func, y0)
        return y  # vh, vz, tangage
    
    def trim_hover(self):
        x = np.zeros(13)
        x[7:10] = 0  # Set the inertial velocity to zero.
        
        def func(y):
            dx, de, theta = y
            x[3:7] = eul2quat([0, theta, 0])
            u = np.array([dx, dx, de, de])
            dq_dt = self.dyn(x, u, np.zeros(3))
            return [dq_dt[7], dq_dt[9], dq_dt[11]]
        
        y0 = np.array([100, 0, 0.1])
        y_sol = fsolve(func, y0)
    
        u = np.array([y_sol[0], y_sol[0], y_sol[1], y_sol[1]])
        return u

    def trim_turn(self, target):
        v, omega = target
        vel = np.array([v, 0, 0])
        rot = np.array([0, 0, omega])
        v_dot = hat(rot)@vel
        print(v_dot)
        
        x = np.zeros(13)
        x[7:10] = vel

        def func(y):
            dx1, dx2, de1, de2, theta, phi = y
            x[3:7] = eul2quat([phi, theta, 0])
            x[10:13] = quatrot(x[3:7], rot)
            u = np.array([dx1, dx2, de1, de2])
            dq_dt = self.dyn(x, u, np.zeros(3)) 
            dv = dq_dt[7:10] - v_dot
            dr = dq_dt[10:13] 
            return np.concatenate([dv, dr])

        y0 = np.array([100, 100, 0, 0, 0, 0])
        y = fsolve(func, y0)
        return y

    def step(self, x, u, w, dt):
        func = lambda t, s: self.dyn(s, u, w)
        sol = solve_ivp(func, (0, dt), x, method='RK45', max_step=0.01)
        return sol.y.T[-1]

    def sim(self, t_span, x0, fctrl, fwind):  #Simulate Function that simulates the movements from time 0 to t seconds.
        func = lambda t, s: self.dyn(s, fctrl(t, s), fwind(t, s))
        def hit_ground(t, s): return -s[2]
        hit_ground.terminal = True
        hit_ground.direction = -1
        sol = solve_ivp(func, t_span, x0, method='RK45', max_step=0.2)
        return sol
