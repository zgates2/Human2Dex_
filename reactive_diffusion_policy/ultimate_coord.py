import matplotlib.pyplot as plt
from spatialmath import SE3

# ============================================
# 定义坐标系变换关系
# ============================================

# 1. htc_ultimate 坐标系 (原点坐标系)
T_htc_ultimate = SE3()

# 2. htc_ultimate -> 法兰坐标系
# 绕自身 z 轴旋转 45 度，再绕新坐标系的 x 轴旋转 -90 度
# 再沿新坐标系的 x, y, z 轴分别平移 -53.2mm, 53.24mm, -41.47mm
# 注意：链式乘法表示 intrinsic/动轴旋转，平移在旋转后的坐标系下进行
T_htc_to_flange = (SE3.Rz(45, unit='deg') * 
                   SE3.Rx(-90, unit='deg') * 
                   SE3.Tx(-0.0532) * 
                   SE3.Ty(0.05324) * 
                   SE3.Tz(-0.04147))

T_flange = T_htc_ultimate * T_htc_to_flange

# 3. 法兰坐标系 -> 力传感器坐标系
# 沿 z 轴移动 30mm (3cm)
T_flange_to_sensor = SE3.Tz(0.03)
T_sensor = T_flange * T_flange_to_sensor

# 4. 法兰坐标系 -> TCP 坐标系
# 沿 z 轴移动 270mm
T_flange_to_tcp = SE3.Tz(0.27)
T_tcp = T_flange * T_flange_to_tcp

# ============================================
# 打印变换矩阵
# ============================================
print("=" * 50)
print("htc_ultimate 坐标系 (原点):")
print(T_htc_ultimate)

print("=" * 50)
print("htc_ultimate -> 法兰 变换矩阵:")
print(T_htc_to_flange)

print("=" * 50)
print("法兰坐标系:")
print(T_flange)

print("=" * 50)
print("力传感器坐标系:")
print(T_sensor)

print("=" * 50)
print("TCP 坐标系:")
print(T_tcp)

# ============================================
# 可视化
# ============================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制各个坐标系
# htc_ultimate 坐标系 (黑色)
T_htc_ultimate.plot(frame='HTC', color='k', ax=ax, length=0.1)

# 法兰坐标系 (蓝色)
T_flange.plot(frame='Flange', color='b', ax=ax, length=0.1)

# 力传感器坐标系 (绿色)
T_sensor.plot(frame='Sensor', color='g', ax=ax, length=0.08)

# TCP 坐标系 (红色)
T_tcp.plot(frame='TCP', color='r', ax=ax, length=0.08)

# 设置绘图范围
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(-0.3, 0.3)

# 设置标签
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Coordinate Frames Visualization\nBlack:HTC_Ultimate | Blue:Flange | Green:Sensor | Red:TCP')

# 添加图例说明
ax.text2D(0.02, 0.98, 
          'Frames:\n'
          '* HTC_Ultimate (Black): Origin\n'
          '* Flange (Blue): Rz(45)->Rx(-90)->Trans(-53.2, 53.24, -41.47)mm\n'
          '* Sensor (Green): Flange + Z(+30mm)\n'
          '* TCP (Red): Flange + Z(+270mm)',
          transform=ax.transAxes, fontsize=9, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
