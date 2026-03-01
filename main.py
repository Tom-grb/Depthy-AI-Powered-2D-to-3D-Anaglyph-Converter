import cv2
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

class Converter2Dto3D:
    def __init__(self, use_gpu=True):
        """
        初始化 2D 转 3D 转换器
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"当前使用设备: {self.device}")
        
        # 针对 4GB 显存优化: 使用 Depth Anything V2 (Small)
        try:
            print("正在加载深度模型 (Depth-Anything-Small)...")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            # 使用 Hugging Face 的 transformers 库加载，避开 GitHub API 限制
            self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
            self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
            self.model_type = "Depth-Anything-HF"
            
        except ImportError:
            print("缺少 transformers 库。尝试使用 pip install transformers 安装。正在回退到 torch.hub...")
            try:
                self.model = torch.hub.load('LiheYoung/depth-anything', 'depth_anything_vits14', source='github')
                self.model_type = "Depth-Anything-Hub"
            except Exception as e:
                 print(f"加载 Depth Anything (Hub) 失败: {e}. 回退到 MiDaS...")
                 self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
                 self.model_type = "MiDaS"
        except Exception as e:
            print(f"加载 Depth Anything (Transformers) 失败: {e}. 回退到 MiDaS...")
            try:
                # 最后的尝试
                 self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
                 self.model_type = "MiDaS"
            except:
                 self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                 self.model_type = "MiDaS"

        self.model.to(self.device).eval()
        
        # 设置预处理变换 (如果不是 HF 模型)
        if self.model_type == "Depth-Anything-Hub":
            self.transform = Compose([
                Resize((518, 518)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.model_type == "MiDaS":
             # MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform

    def estimate_depth(self, img_rgb):
        """
        估算图片的深度图
        """
        h, w = img_rgb.shape[:2]
        
        if self.model_type == "Depth-Anything-HF":
            # Transformers 流程
            # processor 会自动处理 resize 和 normalize
            image_pil = Image.fromarray(img_rgb)
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # 插值回原尺寸
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction
            
        elif self.model_type == "Depth-Anything-Hub":
            # Hub 流程
            image_pil = Image.fromarray(img_rgb)
            img_input = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                depth = self.model(img_input)
                # 插值回原尺寸
                depth = F.interpolate(depth.unsqueeze(1), (h, w), mode="bicubic", align_corners=False).squeeze()
        else:
            # MiDaS 流程
            input_batch = self.transform(img_rgb).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction

        depth = depth.cpu().numpy()
        
        # 归一化深度图 (0-1)
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
        
        # --- 优化重影和立体感 ---
        # 原始: shift = disparity * shift_amount (0 at far, max at near) -> 全部出屏
        # 改进: 引入汇聚点 (Convergence Plane)。
        # 让中间深度的物体在屏幕平面 (0 shift)，近处出屏，远处入屏。
        # 这样人眼对焦更舒服，且能减少单一方向的大位移导致的重影。
        
        convergence_point = 0.5  # 0 (远) - 1 (近). 0.5 表示中间深度在屏幕上
        # 我们希望: 
        # depth < convergence -> shift 符号 A (入屏)
        # depth > convergence -> shift 符号 B (出屏)
        
        # 稍微非线性调整一下深度，让近处的主体更突出
        disparity = torch.pow(disparity, 1.2)
        
        # 计算带汇聚点的位移
        # shift > 0: 出屏
        # shift < 0: 入屏
        # 增加整体 shift_amount 因为现在分摊到入屏和出屏了
        shift_tensor = (disparity - convergence_point) * shift_amount * 1.5 
        
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

def process_file(file_path, output_path, converter, shift_strength=30, save_intermediate=False):
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
        
        if save_intermediate:
            # 保存深度图 (归一化到 0-255)
            depth_uint8 = (depth * 255).astype(np.uint8)
            # 使用伪彩色以便于观察
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
            depth_path = os.path.splitext(output_path)[0] + "_depth.png"
            cv2.imwrite(depth_path, depth_colormap)
            print(f"深度图已保存: {depth_path}")
        
        # 2. 生成立体视图
        left_view, right_view = converter.apply_stereo_shift(img, depth, real_strength)
        
        # 3. 裁剪边缘伪影 (位移后边缘会有拉伸，需要裁掉)
        # 裁剪量 = 位移量 + 2% 的安全边距
        border_x = int(abs(real_strength)) + int(w * 0.02)
        if border_x > 0 and border_x < w//4:
            left_view = left_view[:, border_x:-border_x]
            right_view = right_view[:, border_x:-border_x]
            left_view = cv2.resize(left_view, (w, h))
            right_view = cv2.resize(right_view, (w, h))
            
        if save_intermediate:
            base_name = os.path.splitext(output_path)[0]
            cv2.imwrite(f"{base_name}_left.jpg", left_view)
            cv2.imwrite(f"{base_name}_right.jpg", right_view)
            # 保存并排对比图
            sbs_view = np.hstack((left_view, right_view))
            cv2.imwrite(f"{base_name}_sbs.jpg", sbs_view)
            print(f"中间视图已保存: {base_name}_left/right/sbs.jpg")
        
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
        
        out_width, out_height = width, height
        real_strength = shift_strength * (width / target_scale_width)
            
        # 尝试初始化视频写入器
        # 优先使用 mp4v，因为它不需要额外的 openh264 dll，在 Windows 上通常都能用
        out = None
        current_codec = 'Unknown'
        
        qt_codecs = ['avc1', 'mp4v', 'XVID'] # 尝试列表
        
        for codec in qt_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
                if temp_out.isOpened():
                    out = temp_out
                    current_codec = codec
                    print(f"成功初始化视频编码器: {codec}")
                    break
            except Exception as e:
                continue

        if out is None or not out.isOpened():
             print("错误: 无法创建视频文件。尝试安装 opencv-python-headless 或检查输出路径。")
             cap.release()
             return
        
        pbar = tqdm(total=total_frames, unit="frame")
        
        border_x = int(real_strength) + int(out_width * 0.02)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用深度模型处理每一帧
            depth = converter.estimate_depth(frame_rgb)
            left_view, right_view = converter.apply_stereo_shift(frame, depth, real_strength)
            
            # 裁剪
            if border_x > 0 and border_x < out_width//4:
                 left_view = left_view[:, border_x:-border_x]
                 right_view = right_view[:, border_x:-border_x]
                 left_view = cv2.resize(left_view, (out_width, out_height))
                 right_view = cv2.resize(right_view, (out_width, out_height))
                 
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
    parser.add_argument("--strength", type=float, default=25.0, help="立体分离强度 (建议 30-50)")
    parser.add_argument("--output", help="输出文件路径")
    
    # 保存中间结果
    parser.add_argument("--save-intermediate", action="store_true", help="保存中间结果 (深度图、左右视图)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"未找到输入文件: {args.input}")
        exit(1)
        
    output = args.output
    if not output:
        name, ext = os.path.splitext(args.input)
        output = f"{name}_3d{ext}"
        
    converter = Converter2Dto3D()
    process_file(args.input, output, converter, shift_strength=args.strength, save_intermediate=args.save_intermediate)
