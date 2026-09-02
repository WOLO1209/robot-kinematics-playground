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

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 05：Jacobian 正向速度映射"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=5, suppress=True)
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("关节速度 qdot → Jacobian J → 末端速度 xdot")
print("=" * 72)

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

# 一张教学总览图同时回答三个问题：J 长什么样、输入是什么、输出是什么。
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig.suptitle(EXPERIMENT_TITLE, fontsize=16, fontweight="bold")

ax_heatmap = fig.add_subplot(221)
image = ax_heatmap.imshow(J, cmap="coolwarm", aspect="auto")
ax_heatmap.set_xticks(range(7), [f"q{i + 1}" for i in range(7)])
ax_heatmap.set_yticks(range(6), ["vx", "vy", "vz", "wx", "wy", "wz"])
ax_heatmap.set_title("6×7 Jacobian 热力图")
fig.colorbar(image, ax=ax_heatmap, shrink=0.8, label="速度映射系数")

ax_qdot = fig.add_subplot(222)
ax_qdot.bar(range(1, 8), qdot, color=["#1f77b4"] + ["#aec7e8"] * 6)
ax_qdot.set_xticks(range(1, 8), [f"q{i}" for i in range(1, 8)])
ax_qdot.set_ylabel("关节速度 (rad/s)")
ax_qdot.set_title("输入：只有 q1 运动")
ax_qdot.grid(True, axis="y", alpha=0.3)

ax_xdot = fig.add_subplot(223)
speed_labels = ["vx", "vy", "vz", "wx", "wy", "wz"]
ax_xdot.bar(speed_labels, xdot, color=["#2ca02c"] * 3 + ["#ff7f0e"] * 3)
ax_xdot.axhline(0, color="black", linewidth=0.8)
ax_xdot.set_ylabel("速度分量")
ax_xdot.set_title("输出：末端线速度（绿）与角速度（橙）")
ax_xdot.grid(True, axis="y", alpha=0.3)

ax_arrow = fig.add_subplot(224, projection="3d")
p = robot.fkine(q).t
arrow_scale = 1.5
ax_arrow.scatter(*p, color="#d62728", s=80, label="当前末端")
ax_arrow.quiver(
    p[0], p[1], p[2],
    linear_velocity[0], linear_velocity[1], linear_velocity[2],
    length=arrow_scale, normalize=False, color="#2ca02c", linewidth=3,
    label="末端线速度方向",
)
ax_arrow.set_xlim(p[0] - 0.15, p[0] + 0.15)
ax_arrow.set_ylim(p[1] - 0.15, p[1] + 0.15)
ax_arrow.set_zlim(p[2] - 0.15, p[2] + 0.15)
ax_arrow.set_xlabel("X (m)")
ax_arrow.set_ylabel("Y (m)")
ax_arrow.set_zlabel("Z (m)")
ax_arrow.set_title("末端线速度箭头")
ax_arrow.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show(block=True)
