"""
第 4 节：笛卡尔空间轨迹 Cartesian Trajectory

学习目标：
1. 使用 rtb.ctraj() 直接规划末端位姿轨迹。
2. 对每个末端位姿做 IK，得到可执行的关节角。
3. 理解关节空间规划与笛卡尔空间规划的根本区别。

核心公式：
    T_path = ctraj(T_start, T_target, N)
    q[k] = IK(T_path[k], q0=q[k-1])

运行后应该观察：
末端从起点沿近似直线走向终点。每次 IK 使用上一时刻的关节角
作为初值，因此求得的关节运动通常更连续。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3


EXPERIMENT_TITLE = "实验 04：笛卡尔空间直线轨迹（Cartesian Trajectory）"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("先规划末端直线，再逐点 IK 计算关节角")
print("=" * 72)

robot = rtb.models.DH.Panda()

q_start = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
T_start = robot.fkine(q_start)

# 保持起点朝向不变，只把末端平移一小段可达距离。
p_target = T_start.t + np.array([0.12, 0.10, 0.08])
T_target = SE3.Rt(T_start.R, p_target)

time = np.linspace(0.0, 4.0, 81)
T_trajectory = rtb.ctraj(T_start, T_target, len(time))

# 逐点求 IK。关键技巧：下一点的 q0 使用上一点的解，而不是每次从头猜。
q_trajectory = []
q_previous = q_start.copy()

for step_index, T_step in enumerate(T_trajectory):
    solution = robot.ikine_LM(T_step, q0=q_previous)
    if not solution.success:
        raise RuntimeError(
            f"第 {step_index} 个轨迹点的 IK 未收敛，残差为 {solution.residual}。"
        )
    q_previous = solution.q
    q_trajectory.append(q_previous.copy())

q_trajectory = np.asarray(q_trajectory)

# 用 FK 再次计算真实末端位置，用于验证逐点 IK 的结果。
actual_path = np.array([robot.fkine(q).t for q in q_trajectory])
planned_path = np.array([T_step.t for T_step in T_trajectory])
point_errors = np.linalg.norm(actual_path - planned_path, axis=1)
max_position_error = np.max(point_errors)

print("笛卡尔轨迹点数:", len(time))
print("目标位移 (m):", np.round(T_target.t - T_start.t, 4))
print(f"逐点 IK 的最大位置误差: {max_position_error:.8f} m")
print("\n关节空间：先决定关节怎么动。")
print("笛卡尔空间：先决定末端怎么走，再算关节怎么动。")

fig_robot = plt.figure(figsize=(8, 6))
fig_robot.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig_robot.suptitle(EXPERIMENT_TITLE, fontsize=15, fontweight="bold")
robot.plot(
    q_trajectory,
    backend="pyplot",
    fig=fig_robot,
    dt=time[1] - time[0],
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

# 同时画规划路径和 FK 验证路径。两条线应几乎重合且接近直线。
fig = plt.figure(figsize=(8, 6))
fig.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    planned_path[:, 0],
    planned_path[:, 1],
    planned_path[:, 2],
    "--",
    linewidth=3,
    label="ctraj 规划路径",
)
ax.plot(
    actual_path[:, 0],
    actual_path[:, 1],
    actual_path[:, 2],
    linewidth=1.5,
    label="IK + FK 实际路径",
)
ax.scatter(*actual_path[0], color="green", s=60, label="起点")
ax.scatter(*actual_path[-1], color="red", s=60, label="终点")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"{EXPERIMENT_TITLE}\n规划路径与 IK + FK 实际路径")
ax.legend()
fig.tight_layout()

# 补充关节角与逐点 IK 误差，让学生同时看到“任务空间”和“关节空间”。
fig_detail, (ax_joint, ax_error) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
fig_detail.canvas.manager.set_window_title(EXPERIMENT_TITLE)
for joint_index in range(robot.n):
    ax_joint.plot(
        time,
        np.degrees(q_trajectory[:, joint_index]),
        label=f"q{joint_index + 1}",
    )
ax_joint.set_ylabel("关节角 (deg)")
ax_joint.set_title(f"{EXPERIMENT_TITLE}\n逐点 IK 得到的关节运动")
ax_joint.grid(True, alpha=0.3)
ax_joint.legend(ncol=4, fontsize=9)

ax_error.plot(time, point_errors * 1000, color="#d62728", linewidth=2)
ax_error.axhline(2.0, color="#ff7f0e", linestyle="--", label="2 mm 教学验收线")
ax_error.set_xlabel("时间 (s)")
ax_error.set_ylabel("IK 位置误差 (mm)")
ax_error.grid(True, alpha=0.3)
ax_error.legend()
fig_detail.tight_layout()

plt.show(block=True)
