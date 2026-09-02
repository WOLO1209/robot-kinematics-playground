"""
第 3 节：关节空间轨迹 Joint-space Trajectory

学习目标：
1. 使用 rtb.jtraj() 在两个关节姿态之间生成平滑轨迹。
2. 观察 7 个关节角随时间的变化。
3. 理解关节角平滑并不代表末端走直线。

核心公式：
    q(t) = jtraj(q_start, q_target, t)
    T(t) = FK(q(t))

运行后应该观察：
机械臂从 q_start 平滑运动到 q_target；关节角曲线平滑，
但由 FK 算出的末端三维路径通常是一条弯曲轨迹。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 03：关节空间轨迹规划（Joint-space Trajectory）"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("先规划关节角，再用 FK 观察末端路径是否为直线")
print("=" * 72)

robot = rtb.models.DH.Panda()

q_start = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
q_target = np.array([0.7, 0.2, -0.5, -1.4, 0.5, 1.2, -0.4])

# 4 秒内取 101 个时刻。jtraj 使用五次多项式，让起止速度为零。
time = np.linspace(0.0, 4.0, 101)
trajectory = rtb.jtraj(q_start, q_target, time)
q_trajectory = trajectory.q

# 对每个关节姿态做 FK，得到末端三维路径。
end_effector_path = np.array(
    [robot.fkine(q).t for q in q_trajectory]
)

print(f"轨迹形状: {q_trajectory.shape}（时间点数 × 关节数）")
print("起点关节角 (deg):", np.round(np.degrees(q_trajectory[0]), 2))
print("终点关节角 (deg):", np.round(np.degrees(q_trajectory[-1]), 2))

# block=False 让动画窗口创建后继续执行，最后统一由 plt.show() 保持窗口。
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

# 图 1：7 个关节角随时间变化。显示时转成角度，计算仍是弧度。
fig_joint, ax_joint = plt.subplots(figsize=(9, 5))
fig_joint.canvas.manager.set_window_title(EXPERIMENT_TITLE)
for joint_index in range(robot.n):
    ax_joint.plot(
        time,
        np.degrees(q_trajectory[:, joint_index]),
        label=f"q{joint_index + 1}",
    )
ax_joint.set_title(f"{EXPERIMENT_TITLE}\n7 个关节角平滑变化")
ax_joint.set_xlabel("时间 (s)")
ax_joint.set_ylabel("关节角 (deg)")
ax_joint.grid(True, alpha=0.3)
ax_joint.legend(ncol=4)
fig_joint.tight_layout()

# 图 2：末端的空间路径。它通常不是连接起终点的直线。
fig_path = plt.figure(figsize=(7, 6))
fig_path.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_path = fig_path.add_subplot(111, projection="3d")
ax_path.plot(
    end_effector_path[:, 0],
    end_effector_path[:, 1],
    end_effector_path[:, 2],
    linewidth=2.5,
    label="jtraj 产生的末端路径",
)
# 起点和终点之间的虚线是“理想直线参考”，用来对比实际弯曲路径。
ax_path.plot(
    [end_effector_path[0, 0], end_effector_path[-1, 0]],
    [end_effector_path[0, 1], end_effector_path[-1, 1]],
    [end_effector_path[0, 2], end_effector_path[-1, 2]],
    "--",
    color="#d62728",
    linewidth=2,
    label="起终点直线参考",
)
ax_path.scatter(*end_effector_path[0], color="green", s=60, label="起点")
ax_path.scatter(*end_effector_path[-1], color="red", s=60, label="终点")
ax_path.set_xlabel("X (m)")
ax_path.set_ylabel("Y (m)")
ax_path.set_zlabel("Z (m)")
ax_path.set_title(f"{EXPERIMENT_TITLE}\n关节平滑 ≠ 末端直线")
ax_path.legend()
fig_path.tight_layout()

plt.show(block=True)
