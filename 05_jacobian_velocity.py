"""
第 5 节：Jacobian 与末端速度

学习目标：
1. 使用 robot.jacob0(q) 计算世界坐标系中的 Jacobian。
2. 理解 Panda 的 Jacobian 为什么是 6×7。
3. 从关节速度 qdot 计算末端速度 xdot。

核心公式：
    xdot = J(q) @ qdot

运行后应该观察：
程序只给 q1 设置 0.2 rad/s，其余关节速度为零；终端会输出
Jacobian、末端线速度和末端角速度。
"""

import numpy as np
import roboticstoolbox as rtb


np.set_printoptions(precision=5, suppress=True)

robot = rtb.models.DH.Panda()
q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])

# jacob0 中的 0 表示速度在世界/基坐标系中表达。
J = robot.jacob0(q)

# Panda 有 7 个转动关节，因此 qdot 有 7 个分量。
# 本例只让第 1 个关节转动，速度为 0.2 rad/s。
qdot = np.zeros(robot.n)
qdot[0] = 0.2

# 6 维末端速度的前三项是线速度，后三项是角速度。
xdot = J @ qdot
linear_velocity = xdot[:3]
angular_velocity = xdot[3:]

print("Jacobian J =")
print(J)
print(f"\nJacobian 尺寸: {J.shape}")
print("\n为什么是 6×7？")
print("- 6 行：末端速度有 vx, vy, vz, wx, wy, wz 六个分量。")
print("- 7 列：Panda 有 q1~q7 七个关节速度输入。")
print("- 每一列描述对应关节单位速度对末端速度的瞬时贡献。")

print("\n关节速度 qdot (rad/s):")
print(qdot)
print("\n末端线速度 [vx, vy, vz] (m/s):")
print(linear_velocity)
print("\n末端角速度 [wx, wy, wz] (rad/s):")
print(angular_velocity)

robot.plot(
    q,
    backend="pyplot",
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=True,
)
