"""
第 9 节：Jacobian 连续速度控制——末端沿 X 方向运动

学习目标：
1. 把一次 Jacobian 速度映射扩展为连续控制循环。
2. 理解 Jacobian 是 J(q)，姿态变化后必须重新计算。
3. 使用欧拉积分 q_next = q + qdot * dt 更新关节角。

核心公式：
    J = robot.jacob0(q)
    qdot = pinv(J) @ xdot
    q = q + qdot * dt

运行后应该观察：
末端近似沿 +X 方向匀速移动 0.12 m；Y、Z 基本不变，但 7 个关节中
会有多个关节同时运动。复杂的关节协同可以合成为简单的直线运动。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 09：Jacobian 连续速度控制——末端沿 X 方向运动"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=5, suppress=True)

print("\n" + "=" * 76)
print(EXPERIMENT_TITLE)
print("当前 q -> J(q) -> qdot = pinv(J) @ xdot -> 欧拉积分 -> 新的 q -> FK")
print("=" * 76)

robot = rtb.models.DH.Panda()
q = robot.qr.copy()

# 期望末端只沿世界坐标系 +X 方向移动，后三项为角速度，均设为 0。
xdot = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
dt = 0.05
total_time = 4.0
steps = int(total_time / dt)

q_history = [q.copy()]
qdot_history = []
position_history = [robot.fkine(q).t.copy()]

for _ in range(steps):
    # Jacobian 依赖当前关节角。机械臂一动，下一周期就必须重新计算 J(q)。
    J = robot.jacob0(q)

    # Panda 的 J 是 6×7，不能使用普通矩阵逆；伪逆给出一组最小范数关节速度。
    qdot = np.linalg.pinv(J) @ xdot

    # 欧拉积分：用“速度 × 很短的时间”近似这一小步的关节角变化。
    q = q + qdot * dt

    # FK 把更新后的关节角转换成新的末端位置，用于记录和观察。
    T = robot.fkine(q)
    q_history.append(q.copy())
    qdot_history.append(qdot.copy())
    position_history.append(T.t.copy())

q_history = np.asarray(q_history)              # 形状：(steps + 1, 7)
qdot_history = np.asarray(qdot_history)        # 形状：(steps, 7)
position_history = np.asarray(position_history)  # 形状：(steps + 1, 3)
time_history = np.arange(steps + 1) * dt

p_start = position_history[0]
p_end = position_history[-1]
actual_displacement = p_end - p_start
theoretical_x_displacement = xdot[0] * total_time

print("初始末端位置 (m):", p_start)
print("最终末端位置 (m):", p_end)
print("实际 XYZ 位移 (m):", actual_displacement)
print(f"理论 X 位移: {theoretical_x_displacement:.5f} m")
print("q_history 形状:", q_history.shape)
print("观察重点：末端运动很简单，但多个关节速度会随姿态持续变化。")

# 图 1：三维轨迹。虚线表示理想的纯 X 方向直线。
fig_path = plt.figure(figsize=(7.5, 6))
fig_path.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_path = fig_path.add_subplot(111, projection="3d")
ax_path.plot(*position_history.T, color="#1f77b4", linewidth=2.5, label="实际末端轨迹")
ax_path.plot(
    [p_start[0], p_start[0] + theoretical_x_displacement],
    [p_start[1], p_start[1]],
    [p_start[2], p_start[2]],
    color="#d62728", linestyle="--", linewidth=2, label="理论直线",
)
ax_path.scatter(*p_start, color="#2ca02c", s=70, label="起点")
ax_path.scatter(*p_end, color="#d62728", marker="*", s=120, label="终点")
ax_path.set_ylim(p_start[1] - 0.04, p_start[1] + 0.04)
ax_path.set_zlim(p_start[2] - 0.04, p_start[2] + 0.04)
ax_path.set_xlabel("X Position (m)")
ax_path.set_ylabel("Y Position (m)")
ax_path.set_zlabel("Z Position (m)")
ax_path.set_title(f"{EXPERIMENT_TITLE}\nEnd-effector 3D Path")
ax_path.grid(True, alpha=0.3)
ax_path.legend()
fig_path.tight_layout()

# 图 2：XYZ 随时间变化。理想情况下，只有 X 持续增加。
fig_xyz, ax_xyz = plt.subplots(figsize=(9, 5))
fig_xyz.canvas.manager.set_window_title(EXPERIMENT_TITLE)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
    ax_xyz.plot(
        time_history, position_history[:, axis_index],
        color=colors[axis_index], linewidth=2, label=f"{axis_name} Position",
    )
ax_xyz.set_title(f"{EXPERIMENT_TITLE}\nPosition vs Time")
ax_xyz.set_xlabel("Time (s)")
ax_xyz.set_ylabel("Position (m)")
ax_xyz.grid(True, alpha=0.3)
ax_xyz.legend()
fig_xyz.tight_layout()

# 动画使用二维数组 q_history，每一行是一帧的 7 个关节角。
fig_robot = plt.figure(figsize=(8, 6))
fig_robot.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig_robot.suptitle(EXPERIMENT_TITLE, fontsize=15, fontweight="bold")
robot.plot(
    q_history,
    backend="pyplot",
    fig=fig_robot,
    dt=dt,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

plt.show(block=True)
