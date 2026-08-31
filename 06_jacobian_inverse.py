"""
第 6 节：Jacobian 伪逆与关节速度

学习目标：
1. 从希望的末端速度 xdot 反求关节速度 qdot。
2. 使用 Moore-Penrose 伪逆处理 6×7 的非方阵 Jacobian。
3. 用 J @ qdot 正向验证结果。

核心公式：
    qdot = pinv(J) @ xdot_desired
    xdot_check = J @ qdot

运行后应该观察：
为实现末端沿 +X 方向的 0.1 m/s 速度，7 个关节通常会同时运动；
正向验证速度应接近期望速度。
"""

import numpy as np
import roboticstoolbox as rtb


np.set_printoptions(precision=6, suppress=True)

robot = rtb.models.DH.Panda()
q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
J = robot.jacob0(q)

# [vx, vy, vz, wx, wy, wz]：只要求末端沿世界坐标系 +X 方向运动。
xdot_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

# J 是 6×7 非方阵，不能直接求普通逆矩阵。
# pinv(J) 是 7×6；它给出满足任务速度的最小二范数关节速度解。
qdot = np.linalg.pinv(J) @ xdot_desired

# 再从关节速度正向算回末端速度，检查误差。
xdot_check = J @ qdot
velocity_error = xdot_desired - xdot_check

print(f"Jacobian 尺寸: {J.shape}")
print(f"Jacobian 伪逆尺寸: {np.linalg.pinv(J).shape}")
print("\n期望末端速度 xdot_desired:")
print(xdot_desired)
print("\n伪逆求得的 7 个关节速度 qdot (rad/s):")
print(qdot)
print("\nJ @ qdot 的验证结果:")
print(xdot_check)
print("\n速度误差:")
print(velocity_error)
print(f"误差范数: {np.linalg.norm(velocity_error):.8e}")

print("\nPanda 有 7 个关节，却只需描述 6 维末端速度，因此它是冗余机械臂。")
print("同一个末端速度可能对应多组关节速度；伪逆选择其中最小二范数的一组。")

robot.plot(
    q,
    backend="pyplot",
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=True,
)
