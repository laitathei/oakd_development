from scipy.spatial.transform import Rotation as R
import numpy as np

# q1 q2 q3 q4 tx ty tz from kalibr result.txt
q1 = 0.00322963
q2 = -0.02496303
q3 = -0.00137815
q4 = 0.99968221

tx = -0.09603513
ty = -0.00264404
tz = 0.02017904

dummy_vec = np.array([0.0,0.0,0.0,1.0])

r = R.from_quat([q1,q2,q3,q4])
T_cn_cnm0 = r.as_matrix()
t_vec = np.array([[tx],[ty],[tz]])
T_cn_cnm0 = np.hstack((T_cn_cnm0,t_vec))
T_cn_cnm0 = np.vstack((T_cn_cnm0,dummy_vec))

T_cn_cnm1 = r.as_matrix().T # Inverse of Kalibr result, (transpose for rotation matrix, T'=-R'T)
T_cn_cnm1 = np.hstack((T_cn_cnm1,t_vec))
T_cn_cnm1 = np.vstack((T_cn_cnm1,dummy_vec))

print("T_cn_cnm0:") # T_cn_cnm0 = body_T_cam0
print(T_cn_cnm0)
print("T_cn_cnm1:") # T_cn_cnm0 = body_T_cam1
print(T_cn_cnm1)