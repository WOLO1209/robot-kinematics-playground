"""
第 2 节：逆运动学 Inverse Kinematics (IK)

学习目标：
1. 理解 IK 根据目标末端位置反求关节角 q。
2. 学会检查数值 IK 是否成功。
3. 学会再做一次 FK 验证 IK 的结果。

核心公式：
    T_target -> IK -> q_solution
    q_solution -> FK -> T_actual
    position_error = p_target - p_actual

运行后应该观察：
终端会输出目标位置、求得的关节角、FK 验证位置和位置误差；
图形窗口显示求解后的机械臂姿态。
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3


np.set_printoptions(precision=6, suppress=True)

robot = rtb.models.DH.Panda()

# 数值 IK 需要一个初始猜测。好的初值能让求解更快、更稳定。
q0 = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])

# 本节只要求末端到达这个位置，不约束末端朝向。
p_target = np.array([0.4, 0.2, 0.5])

# IK 接口接收完整 SE(3) 位姿。这里用 q0 的朝向构造一个目标位姿，
# 但下面通过 mask 告诉求解器：只优化 x、y、z 三个位置分量。
R_reference = robot.fkine(q0).R
T_target = SE3.Rt(R_reference, p_target)
position_only_mask = [1, 1, 1, 0, 0, 0]

solution = robot.ikine_LM(
    T_target,
    q0=q0,
    mask=position_only_mask,
)

if not solution.success:
    raise RuntimeError(
        "逆运动学没有收敛。请检查目标是否可达，或尝试更换 q0。"
        f" 求解器残差：{solution.residual}"
    )

q_solution = solution.q

# 必须用 FK 回算，确认 IK 的数值解确实到达目标。
T_actual = robot.fkine(q_solution)
p_actual = T_actual.t
position_error = p_target - p_actual

print("目标位置 p_target (m):")
print(p_target)
print("\nIK 求得的关节角 q_solution (rad):")
print(q_solution)
print("\nIK 求得的关节角 q_solution (deg):")
print(np.degrees(q_solution))
print("\nFK 回算的实际位置 p_actual (m):")
print(p_actual)
print("\n位置误差 p_target - p_actual (m):")
print(position_error)
print(f"\n误差范数: {np.linalg.norm(position_error):.8f} m")

robot.plot(
    q_solution,
    backend="pyplot",
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=True,
)
