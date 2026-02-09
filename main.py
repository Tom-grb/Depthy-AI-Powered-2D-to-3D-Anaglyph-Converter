import cv2
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F

class Converter2Dto3D:
    def __init__(self, use_gpu=True, model_type="DPT_Large"):
        """
        初始化 2D 转 3D 转换器
        
        参数:
        - use_gpu: 是否使用 GPU 加速 (推荐 True)
        - model_type: 深度模型类型
            - "DPT_Large": 质量最高，速度较慢 (推荐)
            - "DPT_Hybrid": 速度和质量平衡
            - "MiDaS_small": 速度最快，质量一般
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"当前使用设备: {self.device}")
        
        print(f"正在加载深度模型 ({model_type})... 首次运行可能需要下载模型。")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载 MiDaS 的预处理变换
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate_depth(self, img_rgb):
        """
        估算图片的深度图
        
        输入: numpy 数组 (H, W, 3) RGB 格式
        输出: 深度图 (H, W)，并归一化到 0-1 之间
        """
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # 辅助函数: 将预测结果调整回原始分辨率
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth = prediction.cpu().numpy()
        
        # 将深度图归一化到 0 到 1 之间
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-8:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
            
        return depth

    def apply_stereo_shift(self, img, depth, shift_amount):
        """
        根据深度图生成左眼和右眼视图
        
        参数:
        - img: 原始图像
        - depth: 深度图
        - shift_amount: 最大位移量 (像素)
        """
        h, w, c = img.shape
        
        # 转换为 Torch 格式以便快速进行网格采样插值
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) # (1, C, H, W)
        
        # 视差张量
        disparity = torch.from_numpy(depth).to(self.device) # (H, W)
        
        # 计算位移量
        shift_tensor = disparity * shift_amount
        
        # 生成归一化坐标网格
        grid_y_norm, grid_x_norm = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        
        # 归一化坐标系下的单个像素宽度
        pixel_w = 2.0 / w
        
        # 逆向映射 (Warping):
        # 左眼视图: 我们希望物体向右移动 (shift > 0)。
        # 因此在生成的新左图中，坐标 x 处的像素应该来自于原图的 (x - shift) 处。
        grid_x_left = grid_x_norm - (shift_tensor * pixel_w) / 2
        
        # 右眼视图: 我们希望物体向左移动。
        # 因此在生成的新右图中，坐标 x 处的像素应该来自于原图的 (x + shift) 处。
        grid_x_right = grid_x_norm + (shift_tensor * pixel_w) / 2
        
        # 堆叠网格
        grid_left = torch.stack((grid_x_left, grid_y_norm), dim=2).unsqueeze(0) # (1, H, W, 2)
        grid_right = torch.stack((grid_x_right, grid_y_norm), dim=2).unsqueeze(0)
        
        start_left = chunk_warping(img_tensor, grid_left)
        start_right = chunk_warping(img_tensor, grid_right)
        
        left_np = start_left.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        right_np = start_right.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        return left_np, right_np

    def make_anaglyph(self, left_img, right_img):
        """
        组合左眼和右眼图像生成红青 3D 图片 (Anaglyph)
        左眼: 红色滤镜 -> 看到红色通道
        右眼: 青色滤镜 -> 看到绿色/蓝色通道
        """
        h, w, c = left_img.shape
        anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 简单红青合成:
        # 最终R = 左眼R
        # 最终G = 右眼G
        # 最终B = 右眼B
        
        # 为了获得更亮、更清晰的效果，我们保持这种简单的通道置换。
        
        anaglyph[:, :, 2] = left_img[:, :, 2] # R 通道 (OpenCV 使用 BGR 顺序，索引 2 是红色)
        anaglyph[:, :, 1] = right_img[:, :, 1] # G 通道
        anaglyph[:, :, 0] = right_img[:, :, 0] # B 通道
        
        return anaglyph

def chunk_warping(img_tensor, grid):
    # 辅助函数: 使用 grid_sample 对输入进行采样
    # 这种方式可以利用 GPU 并行加速，比手动逐像素循环快得多
    return F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

def process_file(file_path, output_path, converter, shift_strength=30):
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    
    # 将强度标准化，基于 1000px 宽度。这样不同分辨率的图片立体感一致。
    target_scale_width = 1000.0
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']:
        print(f"正在处理图片: {file_path}")
        img = cv2.imread(file_path)
        if img is None:
            print(f"读取图片失败: {file_path}")
            return
            
        h, w = img.shape[:2]
        real_strength = shift_strength * (w / target_scale_width)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. 深度估算
        depth = converter.estimate_depth(img_rgb)
        
        # 2. 生成立体视图
        left_view, right_view = converter.apply_stereo_shift(img, depth, real_strength)
        
        # 3. 裁剪边缘伪影 (位移后边缘会有拉伸，需要裁掉)
        # 裁剪量 = 位移量 + 2% 的安全边距
        border_x = int(real_strength) + int(w * 0.02)
        if border_x > 0 and border_x < w//4:
            left_view = left_view[:, border_x:-border_x]
            right_view = right_view[:, border_x:-border_x]
            left_view = cv2.resize(left_view, (w, h))
            right_view = cv2.resize(right_view, (w, h))
        
        # 4. 合成红青图
        anaglyph = converter.make_anaglyph(left_view, right_view)
        
        cv2.imwrite(output_path, anaglyph)
        print(f"图片已保存至: {output_path}")

    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        print(f"正在处理视频: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("无法打开视频文件。")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 优先使用 avc1 (H.264) 编码，兼容性最好。如果失败回退到 mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("提示: avc1 编码器不可用，尝试使用 mp4v 编码...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except Exception as e:
            print(f"编码器初始化异常: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("错误: 无法创建视频文件。可能是路径问题或缺少编码器。尝试安装 opencv-python-headless 或检查输出路径。")
            cap.release()
            return
        
        pbar = tqdm(total=total_frames, unit="frame")
        
        real_strength = shift_strength * (width / target_scale_width)
        border_x = int(real_strength) + int(width * 0.02)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用深度模型处理每一帧
            depth = converter.estimate_depth(frame_rgb)
            left_view, right_view = converter.apply_stereo_shift(frame, depth, real_strength)
            
            # 裁剪
            if border_x > 0 and border_x < width//4:
                 left_view = left_view[:, border_x:-border_x]
                 right_view = right_view[:, border_x:-border_x]
                 left_view = cv2.resize(left_view, (width, height))
                 right_view = cv2.resize(right_view, (width, height))
                 
            anaglyph = converter.make_anaglyph(left_view, right_view)
            
            out.write(anaglyph)
            pbar.update(1)
            
        cap.release()
        out.release()
        pbar.close()
        print(f"视频已保存至: {output_path}")
    else:
        print(f"不支持的文件格式: {ext}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D 图片/视频转红青 3D 工具")
    parser.add_argument("input", help="输入文件路径 (图片或视频)")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--strength", type=float, default=25.0, help="立体分离强度 (默认 25，数值越大立体感越强)")
    parser.add_argument("--model", default="DPT_Large", choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], help="使用的深度模型类型")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"未找到输入文件: {args.input}")
        exit(1)
        
    output = args.output
    if not output:
        name, ext = os.path.splitext(args.input)
        output = f"{name}_3d{ext}"
        
    converter = Converter2Dto3D(model_type=args.model)
    process_file(args.input, output, converter, shift_strength=args.strength)
