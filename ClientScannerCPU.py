import tkinter as tk
import numpy as np
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
# 配置Matplotlib支持中文
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "SimHei"  # 指定黑体（系统自带的中文字体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示方块的问题
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import matplotlib.patches as patches
from spectral.io import envi
from sklearn.pipeline import Pipeline



def fourier_denoise(cube, cutoff_ratio=0.1):
    lines, samples, bands = cube.shape
    denoised_cube = np.zeros_like(cube, dtype=np.float32)

    # 计算截止频率（基于图像尺寸和保留比例）
    cutoff = int(min(lines, samples) * cutoff_ratio)

    for band in range(bands):
        # 1. 提取当前波段数据
        img = cube[:, :, band]

        # 2. 进行傅里叶变换
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)  # 将低频分量移到中心

        # 3. 创建低通滤波器掩码（保留中心低频区域）
        mask = np.zeros((lines, samples), dtype=np.float32)
        center_x, center_y = lines // 2, samples // 2

        # 生成圆形掩码
        y, x = np.ogrid[-center_x:lines - center_x, -center_y:samples - center_y]
        mask_area = x * x + y * y <= cutoff * cutoff
        mask[mask_area] = 1

        # 4. 应用滤波器
        fft_shift_filtered = fft_shift * mask

        # 5. 逆傅里叶变换
        ifft_shift = np.fft.ifftshift(fft_shift_filtered)
        img_denoised = np.fft.ifft2(ifft_shift).real

        # 6. 保存降噪后的波段
        denoised_cube[:, :, band] = img_denoised

    print(f"傅里叶降噪完成，保留低频比例: {cutoff_ratio}，处理波段数: {bands}")
    return denoised_cube


# region=====================read npy files==============================
def save_cube_npy(cube, save_path):
    """将高光谱数据立方体保存为.npy格式（方便后续快速加载）"""
    np.save(save_path, cube)
    print(f"数据已保存为.npy文件：{save_path}")


def read_npy_cube(npy_path):
    """从.npy文件加载高光谱数据立方体"""
    cube = np.load(npy_path, allow_pickle=True)
    print(f"成功加载.npy文件：形状={cube.shape}")
    return cube


# endregion

if __name__ == "__main__":
    # ===========================全局变量定义==============================
    # 存储每个类别的圈选区域坐标（key=类别, value=[(x1,y1,x2,y2), ...]）
    class_regions = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    # 按类别存储采样数量（每个类别独立）
    sample_sizes = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}  # 默认值
    # 高光谱数据全局变量
    cube = None
    lines, samples, bands = 0, 0, 0
    rgb = None
    label = None
    current_class = 1

    # ===========================读取hdr文件==============================
    project_name = "PlTest_emptyname_2025-11-13_02-35-49"
    # project_name = "plastics_training"
    data_dir = r"D:\rubbish\py\plastics_training\PlTest_emptyname_2025-11-13_02-35-49\capture"
    # data_dir = r"D:\rubbish\py\plastics_training\plastics_training\capture"
    hdr_path = data_dir + fr"\{project_name}.hdr"
    raw_path = data_dir + fr"\{project_name}.raw"

    # 加载高光谱数据
    # cube = envi.open(hdr_path, raw_path).load().astype(np.float32)
    cube = envi.open(hdr_path).load().astype(np.float32)
    # 新增：应用傅里叶变换降噪（可调整cutoff_ratio参数控制降噪强度）
    cube = fourier_denoise(cube, cutoff_ratio=0.15)

    lines, samples, bands = cube.shape
    print(f"训练数据维度：行={lines}, 列={samples}, 波段数={bands}")

    # 处理RGB显示（选取指定波段并归一化）
    rgb = cube[:, :, [30, 80, 111]]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # 初始化标注矩阵
    label = np.zeros((lines, samples), dtype=np.int32)


    # ===========================核心功能函数==============================
    def onselect(eclick, erelease):
        """矩形选择标注函数 - 新增：保存圈选区域坐标到class_regions"""
        global label, current_class
        x1, y1 = int(eclick.ydata), int(eclick.xdata)
        x2, y2 = int(erelease.ydata), int(erelease.xdata)

        # 确保坐标范围有效（x1<x2, y1<y2）
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # 标注label矩阵
        label[x_min:x_max, y_min:y_max] = current_class
        # 保存该类别的圈选区域坐标
        class_regions[current_class].append((x_min, y_min, x_max, y_max))

        print(f"Labeled region (行{x_min}:{x_max}, 列{y_min}:{y_max}) as class {current_class}")
        print(f"类别{current_class}已圈选区域数量：{len(class_regions[current_class])}")


    def toggle_class(event):
        """按键切换类别/保存label函数"""
        global current_class
        if event.key == '1':
            current_class = 1
            print("Switched to class 1")
        elif event.key == '2':
            current_class = 2
            print("Switched to class 2")
        elif event.key == '3':
            current_class = 3
            print("Switched to class 3")
        elif event.key == '4':  # 新增类别4支持
            current_class = 4
            print("Switched to class 4")
        elif event.key == '5':  # 新增类别5支持
            current_class = 5
            print("Switched to class 5")
        elif event.key == '6':  # 新增类别6支持
            current_class = 6
            print("Switched to class 6")
        elif event.key == 't':  # 新增：按t键启动SVM训练（标注完成后）
            run_svm_classification()
            print("开始执行SVM分类...")


    def get_user_input():
        """获取每个类别的采样数并更新"""
        global sample_sizes
        # 逐个读取并验证每个类别输入
        try:
            # 类别1
            val1 = int(entry1.get().strip())
            sample_sizes[1] = val1 if val1 > 0 else 20
            # 类别2
            val2 = int(entry2.get().strip())
            sample_sizes[2] = val2 if val2 > 0 else 20
            # 类别3
            val3 = int(entry3.get().strip())
            sample_sizes[3] = val3 if val3 > 0 else 20
            # 类别4
            val4 = int(entry4.get().strip())
            sample_sizes[4] = val4 if val4 > 0 else 20

            val5 = int(entry5.get().strip())
            sample_sizes[5] = val5 if val5 > 0 else 20

            val6 = int(entry6.get().strip())
            sample_sizes[6] = val6 if val6 > 0 else 20

            print(f"\n各类别采样数已更新：")
            print(
                f"类别1：{sample_sizes[1]} | 类别2：{sample_sizes[2]} | 类别3：{sample_sizes[3]} | 类别4：{sample_sizes[4]} | 类别5：{sample_sizes[5]} | 类别6：{sample_sizes[6]}")
        except ValueError:
            print("\n输入错误：请确保所有输入都是正整数！")
            # 恢复默认值
            sample_sizes = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}


    def run_svm_classification():
        start_time = time.time()
        print("开始svm分类")
        """执行SVM分类（核心逻辑：使用圈选坐标+输入的采样数量）"""
        # 检查是否有圈选区域
        for cls in [1, 2, 3, 4, 5, 6]:
            if len(class_regions[cls]) == 0:
                print(f"警告：类别{cls}未圈选任何区域，将不会被计入！")

        # ---------------------- 1. 数据预处理 ----------------------
        scaler = StandardScaler()
        cube_2d = cube.reshape(-1, bands)
        cube_2d_scaled = scaler.fit_transform(cube_2d)

        # ---------------------- 2. 准备训练样本（使用圈选坐标） ----------------------
        np.random.seed(42)
        X_list, y_list = [], []

        # 类别1：使用圈选的坐标范围
        if len(class_regions[1]) > 0:
            for (x1_min, y1_min, x1_max, y1_max) in class_regions[1]:
                rows1 = np.random.choice(range(x1_min, x1_max), sample_sizes[1], replace=False)
                cols1 = np.random.choice(range(y1_min, y1_max), sample_sizes[1], replace=False)
                X1 = cube_2d_scaled[[r * samples + c for r, c in zip(rows1, cols1)]]
                y1 = np.zeros(len(X1), dtype=int)
                X_list.append(X1)
                y_list.append(y1)
        else:
            print("注意你没有圈选PET类型，如有需要，请重启再圈一次")

        # 类别2：使用圈选的坐标范围
        if len(class_regions[2]) > 0:
            for (x2_min, y2_min, x2_max, y2_max) in class_regions[2]:
                rows2 = np.random.choice(range(x2_min, x2_max), sample_sizes[2], replace=False)
                cols2 = np.random.choice(range(y2_min, y2_max), sample_sizes[2], replace=False)
                X2 = cube_2d_scaled[[r * samples + c for r, c in zip(rows2, cols2)]]
                y2 = np.ones(len(X2), dtype=int)
                X_list.append(X2)
                y_list.append(y2)
        else:
            print("注意你没有圈选PE类型，如有需要，请重启再圈一次")

        # 类别3：使用圈选的坐标范围
        if len(class_regions[3]) > 0:
            for (x3_min, y3_min, x3_max, y3_max) in class_regions[3]:
                rows3 = np.random.choice(range(x3_min, x3_max), sample_sizes[3], replace=False)
                cols3 = np.random.choice(range(y3_min, y3_max), sample_sizes[3], replace=False)
                X3 = cube_2d_scaled[[r * samples + c for r, c in zip(rows3, cols3)]]
                y3 = np.full(len(X3), 2, dtype=int)
                X_list.append(X3)
                y_list.append(y3)
        else:
            print("注意你没有圈选PVC类型，如有需要，请重启再圈一次")

        # 类别4：使用圈选的坐标范围
        if len(class_regions[4]) > 0:
            for (x4_min, y4_min, x4_max, y4_max) in class_regions[4]:
                rows4 = np.random.choice(range(x4_min, x4_max), sample_sizes[4], replace=False)
                cols4 = np.random.choice(range(y4_min, y4_max), sample_sizes[4], replace=False)
                X4 = cube_2d_scaled[[r * samples + c for r, c in zip(rows4, cols4)]]
                y4 = np.full(len(X4), 3, dtype=int)
                X_list.append(X4)
                y_list.append(y4)
        else:
            print("注意你没有圈选PP类型，如有需要，请重启再圈一次")

        # 类别5：使用圈选的坐标范围
        if len(class_regions[5]) > 0:
            for (x5_min, y5_min, x5_max, y5_max) in class_regions[5]:
                rows5 = np.random.choice(range(x5_min, x5_max), sample_sizes[5], replace=False)
                cols5 = np.random.choice(range(y5_min, y5_max), sample_sizes[5], replace=False)
                X5 = cube_2d_scaled[[r * samples + c for r, c in zip(rows5, cols5)]]
                y5 = np.full(len(X5), 4, dtype=int)
                X_list.append(X5)
                y_list.append(y5)
        else:
            print("注意你没有圈选HDPE类型，如有需要，请重启再圈一次")

        # 类别6：使用圈选的坐标范围
        if len(class_regions[6]) > 0:
            for (x6_min, y6_min, x6_max, y6_max) in class_regions[6]:
                rows6 = np.random.choice(range(x6_min, x6_max), sample_sizes[6], replace=False)
                cols6 = np.random.choice(range(y6_min, y6_max), sample_sizes[6], replace=False)
                X6 = cube_2d_scaled[[r * samples + c for r, c in zip(rows6, cols6)]]
                y6 = np.full(len(X6), 5, dtype=int)
                X_list.append(X6)
                y_list.append(y6)
        else:
            print("注意你没有圈选OTHER类型，如有需要，请重启再圈一次")

        # 合并训练集
        X_train = np.vstack(X_list)
        y_train = np.hstack(y_list)
        print(f"训练集规模：{X_train.shape}（样本数×波段数），标签数：{len(np.unique(y_train))}类")

        # ---------------------- 3. 训练SVM分类器 ----------------------

        param_grid = {
            'C': [1, 5, 10, 20, 50],  # 增加正则化强度选项
            'gamma': [0.001, 0.01, 0.1, 1, 'scale'],  # 细化核系数
            'kernel': ['rbf', 'poly']  # 新增多项式核函数
        }
        grid_search = GridSearchCV(
            estimator=SVC(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_svm = grid_search.best_estimator_

        print(f"最优SVM参数：{grid_search.best_params_}")
        print(f"交叉验证最优准确率：{grid_search.best_score_:.4f}")
        print("SVM分类器训练完成！")

        # ---------------------- 4. 全图分类预测 ----------------------
        predicted_classes = best_svm.predict(cube_2d_scaled)
        classification_map = predicted_classes.reshape(lines, samples)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"SVM分类完成！总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")

        # ---------------------- 5. 生成伪彩图 ----------------------
        class_names = ["PET", "PE", "PVC", "PP", "HDPE", "OTHER"]
        class_colors = [
            [200, 0, 200], [0, 180, 0], [255, 165, 0], [255, 0, 0], [0, 0, 150], [200, 200, 200]
        ]
        plastic_order = list(zip(class_names, [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in class_colors]))

        # 创建绘图区域
        fig, (ax_map, ax_legend) = plt.subplots(1, 2, figsize=(18, 12), gridspec_kw={"width_ratios": [3, 1]})

        # 绘制分类伪彩图
        cmap = ListedColormap([np.array(color) / 255 for color in class_colors])
        im = ax_map.imshow(classification_map, cmap=cmap, aspect='auto')
        ax_map.set_title('NIR Plastic Classification Map (6 Types)', fontsize=16, pad=20, fontweight='bold')
        ax_map.axis('off')

        # 绘制图例
        n_plastics = len(plastic_order)
        block_height = 1
        total_height = n_plastics * block_height
        ax_legend.set_xlim(0, 3)
        ax_legend.set_ylim(0, total_height)
        ax_legend.axis("off")

        for i, (name, color) in enumerate(plastic_order):
            y_bottom = i * block_height
            y_top = (i + 1) * block_height
            rect = patches.Rectangle((0, y_bottom), width=1, height=block_height, color=color, edgecolor="black")
            ax_legend.add_patch(rect)
            ax_legend.text(1.2, (y_bottom + y_top) / 2, name, fontsize=12, verticalalignment="center")

        ax_legend.text(1.5, total_height + 0.5, "Plastic Type", fontsize=14, fontweight="bold", ha="center")

        plt.tight_layout()
        plt.show()

        X_list.clear()
        y_list.clear()


    # ===========================创建Tkinter窗口==============================
    root = tk.Tk()
    root.title("高光谱图像标注+SVM分类工具")
    root.geometry("1300x800")

    # 创建Matplotlib Figure
    fig = Figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    ax.set_title("1/2/3/4/5/6切换类别 | 鼠标圈选区域 | 输入采样数 | t执行SVM", fontsize=10)

    # 矩形选择器
    rect = RectangleSelector(ax, onselect, interactive=True, useblit=True)

    # 嵌入Matplotlib画布
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # 添加工具栏，目前的作用是看坐标（需要解决使用工具栏后无法回到框选的问题）
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    # 输入框区域（按类别设置采样数）
    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.BOTTOM, pady=10)

    # 类别1采样数
    tk.Label(input_frame, text="类别1采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry1 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry1.insert(0, "20")
    entry1.pack(side=tk.LEFT, padx=2)

    # 类别2采样数
    tk.Label(input_frame, text="类别2采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry2 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry2.insert(0, "20")
    entry2.pack(side=tk.LEFT, padx=2)

    # 类别3采样数
    tk.Label(input_frame, text="类别3采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry3 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry3.insert(0, "20")
    entry3.pack(side=tk.LEFT, padx=2)

    # 类别4采样数
    tk.Label(input_frame, text="类别4采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry4 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry4.insert(0, "20")
    entry4.pack(side=tk.LEFT, padx=2)

    tk.Label(input_frame, text="类别5采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry5 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry5.insert(0, "20")
    entry5.pack(side=tk.LEFT, padx=2)

    tk.Label(input_frame, text="类别6采样数：", font=("SimHei", 12)).pack(side=tk.LEFT, padx=2)
    entry6 = tk.Entry(input_frame, width=8, font=("SimHei", 12))
    entry6.insert(0, "20")
    entry6.pack(side=tk.LEFT, padx=2)

    # 设置采样数按钮（批量更新所有类别）
    tk.Button(input_frame, text="设置所有采样数", command=get_user_input, font=("SimHei", 12)).pack(side=tk.LEFT,
                                                                                                    padx=8)
    # 执行SVM按钮
    tk.Button(input_frame, text="执行SVM分类", command=run_svm_classification, font=("SimHei", 12), bg="#4CAF50",
              fg="white").pack(side=tk.LEFT, padx=8)

    # 布局
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', toggle_class)

    # 启动主循环
    root.mainloop()
