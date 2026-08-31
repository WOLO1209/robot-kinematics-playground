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


# 尽量使用 Windows 常见中文字体；缺少字体时不影响计算。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# DH Panda 是一组连杆参数，不依赖 Swift 或 DAE 网格文件。
robot = rtb.models.DH.Panda()

# 选择一个自然、容易观察的初始姿态。所有数值的单位都是弧度。
q_initial = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
robot.q = q_initial.copy()

# 使用一个已有的 Matplotlib 3D 坐标轴启动 PyPlot 后端。
# 底部预留空间放置 7 个滑块。
fig = plt.figure(figsize=(10, 9))
ax_robot = fig.add_axes([0.08, 0.36, 0.84, 0.58], projection="3d")
env = PyPlot()
env.launch(
    name="01 正运动学 FK 滑块",
    fig=fig,
    ax=ax_robot,
    limits=[-0.8, 0.8, -0.8, 0.8, 0.0, 1.2],
)
env.add(robot)
env.step(0.001)

T_initial = robot.fkine(q_initial)
position_text = fig.text(
    0.08,
    0.945,
    f"末端位置 XYZ (m): {T_initial.t[0]: .3f}, {T_initial.t[1]: .3f}, {T_initial.t[2]: .3f}",
    fontsize=11,
)

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
    position_text.set_text(
        f"末端位置 XYZ (m): {T.t[0]: .3f}, {T.t[1]: .3f}, {T.t[2]: .3f}"
    )

    # PyPlot 根据 robot.q 重画骨架机械臂。
    env.step(0.001)
    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update_robot)

plt.show(block=True)
