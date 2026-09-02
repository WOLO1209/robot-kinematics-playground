"""
第 10 节：Jacobian 多方向连续速度控制

学习目标：
1. 依次给定 +X、+Y、+Z 三段笛卡尔速度。
2. 观察不同 xdot 会产生不同的关节速度组合 qdot。
3. 理解连续运动中每一步都要重新计算 J(q)。

核心公式：
    qdot = pinv(J(q)) @ xdot
    q_next = q + qdot * dt

运行后应该观察：
0–2 s 主要是 X 增加，2–4 s 主要是 Y 增加，4–6 s 主要是 Z 增加；
三段理论位移都约为 0.06 m，但每一段的关节速度组合不同。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 10：Jacobian 多方向连续速度控制"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=5, suppress=True)

print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("相同机器人 + 不同 xdot → 不同 qdot → 不同笛卡尔运动方向")
print("=" * 72)

robot = rtb.models.DH.Panda()
q = robot.qr.copy()
dt = 0.05
segment_time = 2.0
steps_per_segment = int(segment_time / dt)

segments = [
    ("+X", np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])),
]

q_history = [q.copy()]
qdot_history = []
position_history = [robot.fkine(q).t.copy()]
stage_history = []


def move_with_jacobian(q_current, xdot, steps, stage_name):
    """执行一小段连续速度控制；保留公式原貌，便于初学者逐行对应。"""
    for _ in range(steps):
        J = robot.jacob0(q_current)              # 当前姿态的 6×7 Jacobian
        qdot = np.linalg.pinv(J) @ xdot          # qdot = J⁺ @ xdot
        q_current = q_current + qdot * dt        # Euler integration
        T = robot.fkine(q_current)                # FK 得到新的末端位置

        q_history.append(q_current.copy())
        qdot_history.append(qdot.copy())
        position_history.append(T.t.copy())
        stage_history.append(stage_name)
    return q_current


for stage_name, xdot_command in segments:
    print(f"执行阶段 {stage_name}，xdot = {xdot_command}")
    q = move_with_jacobian(q, xdot_command, steps_per_segment, stage_name)

q_history = np.asarray(q_history)
qdot_history = np.asarray(qdot_history)
position_history = np.asarray(position_history)
time_history = np.arange(len(position_history)) * dt

p_start = position_history[0]
p_end = position_history[-1]
actual_displacement = p_end - p_start
theoretical_displacement = 0.03 * segment_time

print("\n初始末端位置 (m):", p_start)
print("最终末端位置 (m):", p_end)
print("实际 XYZ 总位移 (m):", actual_displacement)
print(f"每个方向理论位移: {theoretical_displacement:.5f} m")
print("q_history 形状:", q_history.shape)

# 图 1：三段运动在三维空间中形成折线路径。
fig_path = plt.figure(figsize=(7.5, 6))
fig_path.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_path = fig_path.add_subplot(111, projection="3d")
ax_path.plot(*position_history.T, color="#1f77b4", linewidth=2.5, label="实际 XYZ 路径")
boundary_indices = [0, steps_per_segment, 2 * steps_per_segment, 3 * steps_per_segment]
boundary_labels = ["Start", "X End", "Y End", "Z End"]
boundary_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
for index, label, color in zip(boundary_indices, boundary_labels, boundary_colors):
    ax_path.scatter(*position_history[index], color=color, s=70, label=label)
ax_path.set_xlabel("X Position (m)")
ax_path.set_ylabel("Y Position (m)")
ax_path.set_zlabel("Z Position (m)")
ax_path.set_title(f"{EXPERIMENT_TITLE}\nPiecewise Cartesian Path")
ax_path.grid(True, alpha=0.3)
ax_path.legend(fontsize=8)
fig_path.tight_layout()

# 图 2：用背景色标出三段时间，帮助对应“哪个方向正在运动”。
fig_xyz, ax_xyz = plt.subplots(figsize=(10, 5.5))
fig_xyz.canvas.manager.set_window_title(EXPERIMENT_TITLE)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
    ax_xyz.plot(
        time_history, position_history[:, axis_index],
        color=colors[axis_index], linewidth=2, label=f"{axis_name} Position",
    )
stage_colors = ["#dbeafe", "#ffedd5", "#dcfce7"]
for stage_index, (stage_name, _) in enumerate(segments):
    start_time = stage_index * segment_time
    end_time = (stage_index + 1) * segment_time
    ax_xyz.axvspan(start_time, end_time, color=stage_colors[stage_index], alpha=0.35)
    ax_xyz.text((start_time + end_time) / 2, ax_xyz.get_ylim()[1], stage_name,
                ha="center", va="top", fontweight="bold")
ax_xyz.axvline(segment_time, color="gray", linestyle="--", linewidth=1)
ax_xyz.axvline(2 * segment_time, color="gray", linestyle="--", linewidth=1)
ax_xyz.set_title(f"{EXPERIMENT_TITLE}\nThree Motion Stages")
ax_xyz.set_xlabel("Time (s)")
ax_xyz.set_ylabel("Position (m)")
ax_xyz.grid(True, alpha=0.3)
ax_xyz.legend(ncol=3)
fig_xyz.tight_layout()

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
