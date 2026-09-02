"""
第 11 节：Jacobian 圆周运动 / 时变笛卡尔速度控制

学习目标：
1. 使用随时间变化的末端速度，在 XY 平面形成圆周轨迹。
2. 理解圆的瞬时速度沿切线方向，许多短直线累积后形成曲线。
3. 观察欧拉积分、Jacobian 线性近似和数值误差带来的 drift（漂移）。

核心公式：
    vx = -r * omega * sin(theta)
    vy =  r * omega * cos(theta)
    qdot = pinv(J(q)) @ xdot
    q_next = q + qdot * dt

运行后应该观察：
XY 图接近半径 5 cm 的圆，X/Y 类似正弦和余弦，Z 基本不变；完整一圈后
末端应回到起点附近，但可能存在少量首尾误差。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


EXPERIMENT_TITLE = "实验 11：Jacobian 圆周运动 / 时变笛卡尔速度控制"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=6, suppress=True)

print("\n" + "=" * 78)
print(EXPERIMENT_TITLE)
print("时变切向速度 -> pinv(J(q)) -> qdot -> 连续积分 -> XY 圆周轨迹")
print("=" * 78)

robot = rtb.models.DH.Panda()
q = robot.qr.copy()

dt = 0.02
radius = 0.05
period = 8.0
omega = 2 * np.pi / period
total_time = period
steps = int(total_time / dt)

q_history = [q.copy()]
qdot_history = []
position_history = [robot.fkine(q).t.copy()]
velocity_history = []

for step in range(steps):
    t = step * dt
    theta = omega * t

    # 圆周切向速度。方向随 theta 连续变化，但速度大小近似保持 r * omega。
    vx = -radius * omega * np.sin(theta)
    vy = radius * omega * np.cos(theta)
    xdot = np.array([vx, vy, 0.0, 0.0, 0.0, 0.0])

    # 这里没有使用 ctraj，也没有逐点 IK；全部运动来自速度层 Jacobian 控制。
    J = robot.jacob0(q)
    qdot = np.linalg.pinv(J) @ xdot
    q = q + qdot * dt
    T = robot.fkine(q)

    q_history.append(q.copy())
    qdot_history.append(qdot.copy())
    position_history.append(T.t.copy())
    velocity_history.append(xdot.copy())

q_history = np.asarray(q_history)
qdot_history = np.asarray(qdot_history)
position_history = np.asarray(position_history)
velocity_history = np.asarray(velocity_history)
time_position = np.arange(steps + 1) * dt
time_velocity = np.arange(steps) * dt

p_start = position_history[0]
p_end = position_history[-1]
closure_error = np.linalg.norm(p_end - p_start)
z_drift = np.max(np.abs(position_history[:, 2] - p_start[2]))
max_abs_qdot = np.max(np.abs(qdot_history))

print("初始末端位置 (m):", p_start)
print("最终末端位置 (m):", p_end)
print(f"首尾位置误差: {closure_error:.6f} m ({closure_error * 1000:.3f} mm)")
print(f"最大 Z 漂移: {z_drift:.6f} m ({z_drift * 1000:.3f} mm)")
print(f"最大 |qdot|: {max_abs_qdot:.5f} rad/s")
print("q_history 形状:", q_history.shape)
print("漂移来源：离散时间、Euler integration、Jacobian 线性近似和误差累积。")

# 图 1：三维轨迹，展示圆周基本位于同一 Z 高度。
fig_3d = plt.figure(figsize=(7.5, 6))
fig_3d.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_3d = fig_3d.add_subplot(111, projection="3d")
ax_3d.plot(*position_history.T, color="#1f77b4", linewidth=2.5, label="实际末端轨迹")
ax_3d.scatter(*p_start, color="#2ca02c", s=70, label="起点")
ax_3d.scatter(*p_end, color="#d62728", marker="*", s=120, label="终点")
ax_3d.set_xlabel("X Position (m)")
ax_3d.set_ylabel("Y Position (m)")
ax_3d.set_zlabel("Z Position (m)")
ax_3d.set_title(f"{EXPERIMENT_TITLE}\nEnd-effector 3D Path")
ax_3d.grid(True, alpha=0.3)
ax_3d.legend()
fig_3d.tight_layout()

# 图 2：俯视 XY 平面。axis("equal") 保证圆不会因坐标轴比例不同显示成椭圆。
fig_xy, ax_xy = plt.subplots(figsize=(6.5, 6.5))
fig_xy.canvas.manager.set_window_title(EXPERIMENT_TITLE)
theta_reference = np.linspace(0, 2 * np.pi, 361)
ideal_x = p_start[0] + radius * (np.cos(theta_reference) - 1.0)
ideal_y = p_start[1] + radius * np.sin(theta_reference)
ax_xy.plot(ideal_x, ideal_y, "--", color="#d62728", linewidth=2, label="理论圆")
ax_xy.plot(position_history[:, 0], position_history[:, 1],
           color="#1f77b4", linewidth=2.5, label="实际轨迹")
ax_xy.scatter(p_start[0], p_start[1], color="#2ca02c", s=70, label="起点")
ax_xy.scatter(p_end[0], p_end[1], color="#d62728", marker="*", s=110, label="终点")
ax_xy.axis("equal")
ax_xy.set_xlabel("X Position (m)")
ax_xy.set_ylabel("Y Position (m)")
ax_xy.set_title(f"{EXPERIMENT_TITLE}\nXY Plane (Equal Axis)")
ax_xy.grid(True, alpha=0.3)
ax_xy.legend()
ax_xy.text(
    0.03, 0.03, f"Radius = {radius:.2f} m\nClosure error = {closure_error * 1000:.2f} mm",
    transform=ax_xy.transAxes,
    bbox=dict(boxstyle="round", facecolor="#f0fff4", edgecolor="#2ca02c"),
)
fig_xy.tight_layout()

# 图 3：X/Y 呈周期变化，Z 应保持近似不变。
fig_position, ax_position = plt.subplots(figsize=(9, 5))
fig_position.canvas.manager.set_window_title(EXPERIMENT_TITLE)
for axis_index, (axis_name, color) in enumerate(
    zip(["X", "Y", "Z"], ["#1f77b4", "#ff7f0e", "#2ca02c"])
):
    ax_position.plot(time_position, position_history[:, axis_index],
                     color=color, linewidth=2, label=f"{axis_name} Position")
ax_position.set_title(f"{EXPERIMENT_TITLE}\nPosition vs Time")
ax_position.set_xlabel("Time (s)")
ax_position.set_ylabel("Position (m)")
ax_position.grid(True, alpha=0.3)
ax_position.legend()
fig_position.tight_layout()

# 图 4：Vx/Vy 不断变化，它们组成始终沿圆切线方向的瞬时速度。
fig_velocity, ax_velocity = plt.subplots(figsize=(9, 4.8))
fig_velocity.canvas.manager.set_window_title(EXPERIMENT_TITLE)
ax_velocity.plot(time_velocity, velocity_history[:, 0],
                 color="#1f77b4", linewidth=2, label="Vx")
ax_velocity.plot(time_velocity, velocity_history[:, 1],
                 color="#ff7f0e", linewidth=2, label="Vy")
ax_velocity.axhline(0, color="black", linewidth=0.8)
ax_velocity.set_title(f"{EXPERIMENT_TITLE}\nTime-varying Tangential Velocity")
ax_velocity.set_xlabel("Time (s)")
ax_velocity.set_ylabel("Velocity (m/s)")
ax_velocity.grid(True, alpha=0.3)
ax_velocity.legend()
fig_velocity.tight_layout()

# 适当抽帧，使一整圈动画播放流畅且不会持续过久。
animation_stride = 4
fig_robot = plt.figure(figsize=(8, 6))
fig_robot.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig_robot.suptitle(EXPERIMENT_TITLE, fontsize=15, fontweight="bold")
robot.plot(
    q_history[::animation_stride],
    backend="pyplot",
    fig=fig_robot,
    dt=dt * animation_stride,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

plt.show(block=True)
