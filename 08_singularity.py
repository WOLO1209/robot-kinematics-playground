"""
第 8 节：奇异位形 Singularity

学习目标：
1. 用 Jacobian 奇异值和条件数直观判断速度映射是否敏感。
2. 比较正常姿态与接近奇异姿态。
3. 观察同一个末端速度要求在接近奇异时可能需要很大的关节速度。

核心公式：
    xdot = J @ qdot
    qdot = pinv(J) @ xdot
    condition_number = sigma_max / sigma_min

运行后应该观察：
随着姿态靠近伸展状态，最小奇异值通常变小、条件数变大、
manipulability 下降；在“薄弱方向”产生速度会需要更大的关节速度。
"""

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=6, suppress=True)

robot = rtb.models.DH.Panda()

# q_normal 是一个弯曲、远离完全伸展的普通姿态。
q_normal = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.8])

# 沿着 q_normal -> 接近零姿态的路径采样。零姿态附近是一个直观的
# 伸展区域，Jacobian 往往会变得更敏感。停在 0.995 而非精确零点，
# 是为了展示“接近奇异时速度被放大”，而不是精确奇异时方向完全丢失。
alphas = np.linspace(0.0, 0.995, 120)
q_samples = np.array([(1.0 - alpha) * q_normal for alpha in alphas])

condition_numbers = []
manipulabilities = []
velocity_amplifications = []
weak_directions = []

J_normal = robot.jacob0(q_normal)

for q_sample in q_samples:
    J_sample = robot.jacob0(q_sample)
    U, singular_values, _Vt = np.linalg.svd(J_sample)

    # 条件数越大，表示最大与最小速度增益相差越悬殊。
    condition_numbers.append(np.linalg.cond(J_sample))

    # 六个奇异值的乘积是 Yoshikawa manipulability 的等价计算形式。
    manipulabilities.append(np.prod(singular_values))

    # U 的最后一列是该姿态最难产生的末端速度方向。
    weak_direction = U[:, -1]
    weak_directions.append(weak_direction)
    xdot_test = 0.05 * weak_direction

    qdot_sample = np.linalg.pinv(J_sample) @ xdot_test
    qdot_at_normal = np.linalg.pinv(J_normal) @ xdot_test
    normal_speed = max(np.linalg.norm(qdot_at_normal), 1e-12)
    velocity_amplifications.append(np.linalg.norm(qdot_sample) / normal_speed)

condition_numbers = np.asarray(condition_numbers)
manipulabilities = np.asarray(manipulabilities)
velocity_amplifications = np.asarray(velocity_amplifications)

# 自动选择沿途速度放大最明显的“接近奇异”姿态，避免依赖人工猜姿态。
near_index = int(np.nanargmax(velocity_amplifications[1:])) + 1
q_near_singular = q_samples[near_index]
J_near_singular = robot.jacob0(q_near_singular)

# 对接近奇异姿态最薄弱的末端方向，要求 0.05 的 6D 速度幅值。
xdot_desired = 0.05 * weak_directions[near_index]
qdot_normal = np.linalg.pinv(J_normal) @ xdot_desired
qdot_near_singular = np.linalg.pinv(J_near_singular) @ xdot_desired
xdot_check_normal = J_normal @ qdot_normal
xdot_check_near = J_near_singular @ qdot_near_singular

print("正常姿态 q_normal (deg):")
print(np.degrees(q_normal))
print("\n接近奇异姿态 q_near_singular (deg):")
print(np.degrees(q_near_singular))
print(f"\n正常姿态 Jacobian 条件数: {np.linalg.cond(J_normal):.3e}")
print(f"接近奇异姿态 Jacobian 条件数: {np.linalg.cond(J_near_singular):.3e}")
print("\n相同的期望末端速度 xdot_desired:")
print(xdot_desired)
print("\n正常姿态所需 qdot (rad/s):")
print(qdot_normal)
print("接近奇异姿态所需 qdot (rad/s):")
print(qdot_near_singular)
print(f"\n正常姿态关节速度范数: {np.linalg.norm(qdot_normal):.4f}")
print(f"接近奇异姿态关节速度范数: {np.linalg.norm(qdot_near_singular):.4f}")
print(
    "速度放大倍数: "
    f"{np.linalg.norm(qdot_near_singular) / max(np.linalg.norm(qdot_normal), 1e-12):.1f}×"
)
print("\n正向验证（正常姿态）:", xdot_check_normal)
print("正向验证（接近奇异）:", xdot_check_near)

# 条件数和 manipulability 的量纲与范围不同，因此使用两个纵轴。
fig, ax_condition = plt.subplots(figsize=(9, 5))
ax_condition.semilogy(
    alphas,
    condition_numbers,
    color="tab:red",
    linewidth=2,
    label="Jacobian 条件数",
)
ax_condition.axvline(
    alphas[near_index],
    color="gray",
    linestyle="--",
    label="选中的近奇异姿态",
)
ax_condition.set_xlabel("从正常姿态走向伸展姿态的比例 α")
ax_condition.set_ylabel("条件数（对数坐标）", color="tab:red")
ax_condition.tick_params(axis="y", labelcolor="tab:red")
ax_condition.grid(True, which="both", alpha=0.3)

ax_manipulability = ax_condition.twinx()
ax_manipulability.plot(
    alphas,
    manipulabilities,
    color="tab:blue",
    linewidth=2,
    label="Manipulability",
)
ax_manipulability.set_ylabel("Manipulability", color="tab:blue")
ax_manipulability.tick_params(axis="y", labelcolor="tab:blue")
ax_condition.set_title("接近奇异位形：条件数上升、可操作度下降")

lines_1, labels_1 = ax_condition.get_legend_handles_labels()
lines_2, labels_2 = ax_manipulability.get_legend_handles_labels()
ax_condition.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
fig.tight_layout()

# 分别打开正常姿态和近奇异姿态窗口，便于直观比较机械臂形状。
robot.plot(
    q_normal,
    backend="pyplot",
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)
robot_near = rtb.models.DH.Panda()
robot_near.plot(
    q_near_singular,
    backend="pyplot",
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
    block=False,
)

plt.show(block=True)
