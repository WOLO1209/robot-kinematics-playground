# Robot Kinematics Playground

一个面向初学者的机械臂运动学实验场。项目使用 Franka Emika Panda 的 DH 模型，通过 8 个可以独立运行的 Python 示例，把抽象公式变成可观察的数字、曲线和动画。

项目只使用 Matplotlib / PyPlot 绘制“骨架式”机械臂，不加载 Swift 的 Collada/DAE 网格，因此更适合 Windows 桌面环境。

## 适合人群

- 第一次接触机械臂、机器人学或 Robotics Toolbox for Python 的学习者
- 已经会运行基础 Python，希望直观理解运动学概念的学生
- 需要一套可修改、可演示的中文教学代码的教师和助教

你不需要提前学过矩阵推导，但建议了解 NumPy 数组和最基本的线性代数。

## 学习目标

完成本项目后，你应该能够：

1. 区分关节空间（Joint Space）与笛卡尔空间（Cartesian Space）。
2. 理解正运动学 Forward Kinematics (FK) 与逆运动学 Inverse Kinematics (IK) 的输入输出。
3. 解释关节空间轨迹 `jtraj` 和笛卡尔轨迹 `ctraj` 的区别。
4. 使用 Jacobian 在关节速度与末端速度之间转换。
5. 理解 7 自由度机械臂为什么需要 Jacobian 伪逆（Pseudo-inverse）。
6. 看懂最基础的闭环位置控制流程。
7. 直观认识奇异位形（Singularity）及其带来的数值问题。

## 核心概念关系

```text
关节角 q
   ↓ Forward Kinematics (FK)
末端位姿 T

末端位姿 T
   ↓ Inverse Kinematics (IK)
关节角 q

关节速度 qdot
   ↓ Jacobian J
末端速度 xdot

末端速度 xdot
   ↓ Jacobian pseudo-inverse J⁺
关节速度 qdot
```

这里的末端位姿 `T` 同时包含位置和姿态；末端速度 `xdot` 包含 3 个线速度分量和 3 个角速度分量。

## 环境要求

- Windows 10/11（其他桌面系统通常也可运行）
- Python 3.12
- 可显示 Matplotlib 图形窗口的桌面环境

主要依赖：

- `roboticstoolbox-python`
- `spatialmath-python`
- `numpy`
- `matplotlib`

## Windows 安装步骤

打开 PowerShell，进入项目目录，然后执行：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果电脑上有多个 Python 版本，请先运行 `python --version`，确认显示的是 Python 3.12。

本项目没有安装 Swift 可视化扩展，也不需要浏览器加载机器人网格。第一次安装 Robotics Toolbox 可能需要一些时间。

## 如何运行

保持虚拟环境已激活，每次只运行一个示例：

```powershell
python 01_fk_slider.py
python 02_inverse_kinematics.py
python 03_joint_trajectory.py
python 04_cartesian_trajectory.py
python 05_jacobian_velocity.py
python 06_jacobian_inverse.py
python 07_closed_loop_control.py
python 08_singularity.py
```

图形示例会打开一个或多个 Matplotlib 窗口。先查看图形和终端输出，关闭当前窗口后再运行下一章。

## 推荐学习顺序

| 章节 | 核心主题 | 学完应该理解什么 |
| --- | --- | --- |
| 01 | 正运动学 FK | 改变关节角会如何改变末端位置；`q → FK → T` |
| 02 | 逆运动学 IK | 已知目标位置如何反求关节角，以及为什么要用 FK 验证 |
| 03 | 关节空间轨迹 | `jtraj` 让关节平滑变化，但末端路径不保证为直线 |
| 04 | 笛卡尔空间轨迹 | `ctraj` 先规划末端路径，再逐点通过 IK 求关节角 |
| 05 | Jacobian 正向映射 | 如何用 `xdot = J @ qdot` 计算末端瞬时速度 |
| 06 | Jacobian 伪逆 | 如何用 `qdot = J⁺ @ xdot` 反求冗余机械臂的关节速度 |
| 07 | 闭环位置控制 | “读取—比较—计算—更新—反馈”如何让末端逼近目标 |
| 08 | 奇异位形 | 为什么接近奇异时条件数变差、关节速度可能急剧增大 |

## `jtraj` 与 `ctraj`

```text
jtraj:
先规划关节角
-> 关节运动平滑
-> 末端路径不一定直

ctraj:
先规划末端路径
-> 每个路径点做 IK
-> 末端可以走直线
```

两者不是谁“更好”，而是规划对象不同。关节空间轨迹通常更直接；当末端必须沿指定路径移动时，笛卡尔空间轨迹更符合任务需求。

## 每个示例的观察方法

### 01 — FK 滑块

拖动 7 个滑块，观察机械臂和末端 XYZ 的实时变化。内部角度始终用弧度计算，界面使用角度方便阅读。

### 02 — IK 与 FK 验证

查看终端中的目标位置、IK 关节角、FK 回算位置和误差。IK 是数值问题，必须检查 `success`，并用 FK 验证结果。

### 03 — 关节空间轨迹

先查看机械臂动画，再看 7 条关节角曲线和末端三维路径。注意末端曲线通常不是空间直线。

### 04 — 笛卡尔空间轨迹

观察 `ctraj` 产生的末端直线路径。程序逐点求 IK，并把上一点的关节角作为下一点的初值，以提高解的连续性。

### 05 — Jacobian 速度

只让 q1 运动，查看 6×7 Jacobian 如何把 7 个关节速度映射成 6 个末端速度分量。

### 06 — Jacobian 伪逆

指定末端沿 x 方向运动，使用伪逆求 7 个关节速度，再正向验证得到的末端速度。

### 07 — 闭环位置控制

查看末端 XYZ 收敛到目标值、误差下降，以及三维运动轨迹。速度限制可防止一次更新过大。

### 08 — 奇异位形

比较正常姿态与接近奇异姿态的奇异值、条件数和所需关节速度。条件数越大，速度映射通常越敏感。

## 常见问题

**图形窗口没有出现？**  请确认是在本地桌面终端运行，而不是无图形界面的服务器；也可以检查 Matplotlib 是否能独立显示窗口。

**IK 提示失败？**  数值 IK 对目标、初值和版本有一定敏感性。示例已选用较容易收敛的目标；如修改目标，请确保它在机械臂可达范围内。

**中文字体出现方框？**  代码会尝试使用 Windows 常见中文字体。若系统没有这些字体，图形仍可运行，但标题中的中文可能无法正确显示。

## 进一步实验

- 修改目标位置，观察哪些目标不可达。
- 增加或减少轨迹采样点，比较动画平滑程度。
- 修改闭环增益 `K`、时间步长 `dt` 和速度上限。
- 在接近奇异位形时尝试阻尼最小二乘（Damped Least Squares），并与普通伪逆比较。

## License

本项目使用 [MIT License](LICENSE)。欢迎用于学习、教学和二次开发。
