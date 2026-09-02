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

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 06：Jacobian 伪逆与冗余关节速度"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=6, suppress=True)
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("期望末端速度 xdot → Jacobian 伪逆 J+ → 关节速度 qdot")
print("=" * 72)

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

fig_robot = plt.figure(figsize=(8, 6))
fig_robot.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig_robot.suptitle(EXPERIMENT_TITLE, fontsize=15, fontweight="bold")
robot.plot(
    q,
    backend="pyplot",
    fig=fig_robot,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

fig, (ax_qdot, ax_check) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig.suptitle(EXPERIMENT_TITLE, fontsize=16, fontweight="bold")

colors = ["#1f77b4" if value >= 0 else "#d62728" for value in qdot]
ax_qdot.bar([f"q{i}" for i in range(1, 8)], qdot, color=colors)
ax_qdot.axhline(0, color="black", linewidth=0.8)
ax_qdot.set_ylabel("关节速度 (rad/s)")
ax_qdot.set_title("伪逆给出的 7 个关节速度\n蓝色为正，红色为负")
ax_qdot.grid(True, axis="y", alpha=0.3)

component_names = ["vx", "vy", "vz", "wx", "wy", "wz"]
x = np.arange(6)
width = 0.36
ax_check.bar(x - width / 2, xdot_desired, width, label="期望 xdot", color="#ff7f0e")
ax_check.bar(x + width / 2, xdot_check, width, label="J @ qdot 验证", color="#2ca02c")
ax_check.set_xticks(x, component_names)
ax_check.set_ylabel("末端速度分量")
ax_check.set_title("正向验证：两组柱应几乎重合")
ax_check.grid(True, axis="y", alpha=0.3)
ax_check.legend()
ax_check.text(
    0.03, 0.93,
    f"验证误差 = {np.linalg.norm(velocity_error):.2e}",
    transform=ax_check.transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="#f0fff4", edgecolor="#2ca02c"),
)

fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show(block=True)
