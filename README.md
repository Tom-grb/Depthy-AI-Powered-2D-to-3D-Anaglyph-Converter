# 2D 转 3D 转换器 (红蓝立体 / Red-Cyan Anaglyph)

此工具使用 AI 深度估计 (MiDaS) 将标准 2D 图像和视频转换为红蓝 3D 立体格式。

## 环境要求 (Requirements)

1.  Python 3.8+
2.  **设置虚拟环境 (推荐):**

    ### Windows (PowerShell)
    如果你遇到 "无法加载文件...因为在此系统上禁止运行脚本" (UnauthorizedAccess) 的错误，请先运行权限策略命令。

    ```powershell
    # 1. 创建虚拟环境
    python -m venv venv
    
    # 2. 允许运行脚本 (解决权限错误)
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    
    # 3. 激活环境 (注意路径中的 .ps1)
    .\venv\Scripts\Activate.ps1
    ```
    *成功激活后，命令行前面会出现 `(venv)` 字样。*

    ### Linux / macOS
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    注意：如果你有 NVIDIA 显卡，建议安装支持 CUDA 的 PyTorch。如果默认安装的版本不支持 GPU，请访问 [pytorch.org](https://pytorch.org/get-started/locally/) 获取特定的安装命令。
    例如：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 使用方法 (Usage)

确保你已经激活了虚拟环境 (命令行前显示 `(venv)`)。

### 处理图片
```bash
python main.py path/to/image.jpg
```

### 处理视频
```bash
python main.py path/to/video.mp4
```

### 选项 (Options)
*   `--strength`: 调整 3D 效果强度 (深度)。默认值为 25。
    *   **改进算法说明**: 新版本引入了“零视差平面”算法，让中间物体位于屏幕上，近处出屏，远处入屏，大幅减少重影并提升观看舒适度。
    *   建议范围: 30 - 60。
    ```bash
    python main.py image.jpg --strength 45
    ```
*   `--save-intermediate`: 保存中间处理结果，包括深度图、左右单眼各视图、以及并排 (Side-by-Side) 视图。
    ```bash
    python main.py image.jpg --save-intermediate
    ```

## 输出 (Output)
输出文件将保存在同一文件夹中，文件名后附加 `_3d` (例如 `image_3d.jpg`)。
