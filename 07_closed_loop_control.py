"""
第 7 节：基础闭环位置控制 Closed-loop Position Control

学习目标：
1. 实现最简单的比例位置反馈控制器。
2. 使用 Jacobian 伪逆把末端速度转换为关节速度。
3. 观察位置误差在反复反馈中逐渐减小。

核心公式：
    error = p_target - p
    v = K * error
    xdot = [v, 0, 0, 0]
    qdot = pinv(J) @ xdot
    q = q + qdot * dt

运行后应该观察：
XYZ 曲线逐渐接近目标值，位置误差下降到阈值以下，末端沿一条
连续路径到达目标。流程会不断重复“读取 -> 误差 -> 速度 -> 更新 -> 反馈”。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 07：Jacobian 伪逆闭环位置控制"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("读取位置 → 计算误差 → 生成速度 → 更新关节 → 再反馈")
print("=" * 72)

robot = rtb.models.DH.Panda()

q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
p_start = robot.fkine(q).t

# 选择一个离初始位置不远的目标，减少超出工作空间或遇到奇异性的概率。
p_target = p_start + np.array([0.10, 0.08, 0.06])

K = 1.5                    # 比例增益：误差越大，命令速度越大
dt = 0.02                  # 离散控制周期，单位为秒
max_joint_speed = 0.8      # 每个关节的最大速度，单位 rad/s
position_tolerance = 0.002 # 到目标 2 mm 以内就停止
max_steps = 600            # 安全上限，避免程序无限循环

time_history = []
position_history = []
error_history = []
q_history = []

for step in range(max_steps):
    # 1) 读取当前末端位置。
    T = robot.fkine(q)
    p = T.t

    # 2) 计算位置误差。
    error = p_target - p
    error_norm = np.linalg.norm(error)

    time_history.append(step * dt)
    position_history.append(p.copy())
    error_history.append(error_norm)
    q_history.append(q.copy())

    if error_norm < position_tolerance:
        print(f"第 {step} 步到达误差阈值。")
        break

    # 3) 比例控制器产生末端线速度。后三项为零，表示希望保持末端朝向。
    linear_velocity = K * error
    xdot = np.r_[linear_velocity, np.zeros(3)]

    # 4) 用当前姿态的 Jacobian 伪逆反求关节速度。
    J = robot.jacob0(q)
    qdot = np.linalg.pinv(J) @ xdot

    # 5) 限制每个关节速度，避免接近奇异位置时出现过大的数值。
    qdot = np.clip(qdot, -max_joint_speed, max_joint_speed)

    # 6) 欧拉积分更新关节角，下一轮会重新读取并反馈。
    q = q + qdot * dt
else:
    print("达到最大步数，尚未进入误差阈值。可检查目标、增益或速度上限。")

time_history = np.asarray(time_history)
position_history = np.asarray(position_history)
error_history = np.asarray(error_history)
q_history = np.asarray(q_history)

print("初始位置 (m):", np.round(p_start, 5))
print("目标位置 (m):", np.round(p_target, 5))
print("最终位置 (m):", np.round(position_history[-1], 5))
print(f"最终位置误差: {error_history[-1]:.6f} m")
print("\n闭环流程：读取当前位置 -> 计算误差 -> 算速度 -> 更新 -> 再反馈")

# 图 1：X、Y、Z 随时间变化，同时画出各自目标值。
fig_xyz, ax_xyz = plt.subplots(figsize=(9, 5))
fig_xyz.canvas.manager.set_window_title(EXPERIMENT_TITLE)
axis_names = ["X", "Y", "Z"]
for axis_index, axis_name in enumerate(axis_names):
    ax_xyz.plot(
        time_history,
        position_history[:, axis_index],
        label=f"{axis_name} 实际值",
    )
    ax_xyz.axhline(
        p_target[axis_index],
        linestyle="--",
        alpha=0.6,
        label=f"{axis_name} 目标值",
    )
ax_xyz.set_title(f"{EXPERIMENT_TITLE}\n末端 XYZ 随时间收敛到目标")
ax_xyz.set_xlabel("时间 (s)")
ax_xyz.set_ylabel("位置 (m)")
ax_xyz.grid(True, alpha=0.3)
ax_xyz.legend(ncol=3, fontsize=9)
fig_xyz.tight_layout()

# 图 2：误差范数。使用对数纵轴，更容易看到误差逐步下降。
fig_error, ax_error = plt.subplots(figsize=(8, 4.5))
fig_error.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_error.semilogy(time_history, error_history, linewidth=2)
ax_error.axhline(position_tolerance, color="red", linestyle="--", label="停止阈值")
# 用背景色区分闭环过程的三个阶段，帮助初学者读懂收敛过程。
total_time = time_history[-1]
ax_error.axvspan(0, total_time * 0.25, color="#fff3cd", alpha=0.35, label="快速逼近")
ax_error.axvspan(total_time * 0.25, total_time * 0.75, color="#dbeafe", alpha=0.25, label="持续修正")
ax_error.axvspan(total_time * 0.75, total_time, color="#dcfce7", alpha=0.30, label="进入目标区")
ax_error.set_title(f"{EXPERIMENT_TITLE}\n位置误差在反馈中持续下降")
ax_error.set_xlabel("时间 (s)")
ax_error.set_ylabel("位置误差范数 (m)")
ax_error.grid(True, which="both", alpha=0.3)
ax_error.legend()
fig_error.tight_layout()

# 图 3：末端三维路径。
fig_path = plt.figure(figsize=(7, 6))
fig_path.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_path = fig_path.add_subplot(111, projection="3d")
ax_path.plot(
    position_history[:, 0],
    position_history[:, 1],
    position_history[:, 2],
    linewidth=2.5,
    label="闭环控制路径",
)
ax_path.scatter(*position_history[0], color="green", s=60, label="起点")
ax_path.scatter(*p_target, color="red", marker="*", s=120, label="目标")
ax_path.set_xlabel("X (m)")
ax_path.set_ylabel("Y (m)")
ax_path.set_zlabel("Z (m)")
ax_path.set_title(f"{EXPERIMENT_TITLE}\n末端三维轨迹")
ax_path.text2D(
    0.03, 0.03,
    f"初始误差：{error_history[0] * 1000:.1f} mm\n"
    f"最终误差：{error_history[-1] * 1000:.2f} mm\n"
    f"控制步数：{len(error_history) - 1}",
    transform=ax_path.transAxes,
    bbox=dict(boxstyle="round", facecolor="#f0fff4", edgecolor="#2ca02c"),
)
ax_path.legend()
fig_path.tight_layout()

# 动画只取部分控制点，既保留运动过程，也避免播放过慢。
animation_stride = max(1, len(q_history) // 100)
fig_robot = plt.figure(figsize=(8, 6))
fig_robot.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig_robot.suptitle(EXPERIMENT_TITLE, fontsize=15, fontweight="bold")
robot.plot(
    q_history[::animation_stride],
    backend="pyplot",
    fig=fig_robot,
    dt=0.04,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

plt.show(block=True)
