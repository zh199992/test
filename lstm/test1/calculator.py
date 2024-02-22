
import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 2, 3, 4, 5]

# 绘制图形
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# 添加图例标注
plt.legend()

# 显示图形
plt.show()