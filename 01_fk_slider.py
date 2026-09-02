"""
第 1 节：正运动学 Forward Kinematics (FK)

学习目标：
1. 理解 7 个关节角组成向量 q。
2. 理解正运动学把 q 映射为末端位姿 T。
3. 通过滑块观察关节角变化如何影响末端 XYZ。

核心公式：
    q -> FK -> T
    T = robot.fkine(q)

运行后应该观察：
拖动 q1~q7 的滑块，机械臂姿态会更新，左上角的末端 XYZ 也会变化。
界面显示角度，机器人内部计算仍使用弧度。
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot


EXPERIMENT_TITLE = "实验 01：正运动学交互滑块（Forward Kinematics）"

# 尽量使用 Windows 常见中文字体；缺少字体时不影响计算。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# DH Panda 是一组连杆参数，不依赖 Swift 或 DAE 网格文件。
robot = rtb.models.DH.Panda()
print("\n" + "=" * 72)
print(EXPERIMENT_TITLE)
print("拖动 q1~q7，观察：关节角 q → FK → 末端位姿 T")
print("=" * 72)

# 选择一个自然、容易观察的初始姿态。所有数值的单位都是弧度。
q_initial = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
robot.q = q_initial.copy()

# 使用一个已有的 Matplotlib 3D 坐标轴启动 PyPlot 后端。
# 底部预留空间放置 7 个滑块。
fig = plt.figure(figsize=(12, 9))
fig.canvas.manager.set_window_title(EXPERIMENT_TITLE)
fig.suptitle(EXPERIMENT_TITLE, fontsize=16, fontweight="bold", y=0.98)
ax_robot = fig.add_axes([0.04, 0.36, 0.62, 0.56], projection="3d")
env = PyPlot()
env.launch(
    name=EXPERIMENT_TITLE,
    fig=fig,
    ax=ax_robot,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
)
env.add(robot)
env.step(0.001)

T_initial = robot.fkine(q_initial)

# 右侧教学信息面板同步显示 q、XYZ 和齐次变换矩阵 T。
ax_info = fig.add_axes([0.69, 0.39, 0.28, 0.50])
ax_info.set_facecolor("#f5f7fa")
ax_info.set_xticks([])
ax_info.set_yticks([])
for spine in ax_info.spines.values():
    spine.set_color("#d0d7de")
info_text = ax_info.text(0.05, 0.95, "", va="top", fontsize=10)


def format_information(q, T):
    """把当前数值排成适合初学者阅读的信息面板。"""
    q_deg = np.degrees(q)
    q_lines = "\n".join(
        f"  q{i + 1} = {angle:7.2f}°" for i, angle in enumerate(q_deg)
    )
    matrix_text = np.array2string(T.A, precision=3, suppress_small=True)
    return (
        "关节角 q（显示为度）\n"
        f"{q_lines}\n\n"
        "末端位置 XYZ（米）\n"
        f"  x = {T.t[0]: .3f}\n  y = {T.t[1]: .3f}\n  z = {T.t[2]: .3f}\n\n"
        "末端位姿矩阵 T\n"
        f"{matrix_text}\n\n"
        "观察：q 改变后，FK 会立即更新 T。"
    )


info_text.set_text(format_information(q_initial, T_initial))

# 每个关节的上下限来自 Panda 模型。Slider 为了便于阅读使用“度”。
sliders = []
for joint_index in range(robot.n):
    slider_ax = fig.add_axes([0.17, 0.29 - 0.04 * joint_index, 0.70, 0.025])
    q_min_deg = np.degrees(robot.qlim[0, joint_index])
    q_max_deg = np.degrees(robot.qlim[1, joint_index])
    slider = Slider(
        ax=slider_ax,
        label=f"q{joint_index + 1} (°)",
        valmin=q_min_deg,
        valmax=q_max_deg,
        valinit=np.degrees(q_initial[joint_index]),
        valstep=0.5,
    )
    sliders.append(slider)


def update_robot(_value):
    """读取 7 个滑块，做一次 FK，并刷新机械臂和文字。"""
    # Slider 给出角度，先转换回弧度再交给机器人模型。
    q = np.radians([slider.val for slider in sliders])
    robot.q = q

    # FK 的结果是 SE(3) 位姿；其中 .t 是末端位置 [x, y, z]。
    T = robot.fkine(q)
    info_text.set_text(format_information(q, T))

    # PyPlot 根据 robot.q 重画骨架机械臂。
    env.step(0.001)
    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update_robot)

plt.show(block=True)
