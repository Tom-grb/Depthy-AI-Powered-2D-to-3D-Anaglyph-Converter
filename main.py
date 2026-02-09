import cv2
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from ai_utils import ImageEnhancer
from fast_enhance import FastEnhancer

class Converter2Dto3D:
    def __init__(self, use_gpu=True, model_type="DPT_Large", enhance_mode=None):
        """
        初始化 2D 转 3D 转换器
        
        参数:
        - enhance_mode: "hq" (Real-ESRGAN, 慢但清晰), "fast" (FSRCNN, 快), 或 None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"当前使用设备: {self.device}")
        
        # 初始化画质增强器
        self.enhancer = None
        if enhance_mode == "hq":
            print("初始化高清画质增强 (Real-ESRGAN)... 速度较慢") 
            self.enhancer = ImageEnhancer(use_gpu=use_gpu)
        elif enhance_mode == "fast":
            print("初始化快速画质增强 (FSRCNN)... 速度较快")
            self.enhancer = FastEnhancer(use_gpu=use_gpu)
            self.enhancer.load_model()
        
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

def process_file(file_path, output_path, converter, shift_strength=30, upscale_factor=2):
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
            
        # 0. 画质增强
        if converter.enhancer:
            try:
                # FastEnhancer 不接受 out_scale 参数，默认 x2
                if hasattr(converter.enhancer, 'model_name') and converter.enhancer.model_name == 'fsrcnn':
                   img = converter.enhancer.enhance(img)
                else:
                   print(f"正在应用 AI 画质增强 (x{upscale_factor})... 这可能需要一些时间")
                   img = converter.enhancer.enhance(img, out_scale=upscale_factor)
            except Exception as e:
                print(f"画质增强失败: {e}")
                # 继续处理原图
            print(f"增强后分辨率: {img.shape[1]}x{img.shape[0]}")
            
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
        
        # 如果启用了增强，计算新的分辨率
        out_width, out_height = width, height
        if converter.enhancer:
            is_fast = hasattr(converter.enhancer, 'model_name') and converter.enhancer.model_name == 'fsrcnn'
            if is_fast:
                print("正在使用 FSRCNN 快速增强 (固定 x2 倍)...")
                upscale_factor = 2.0
            else:
                # 警告: 视频逐帧超分非常慢
                print("警告: 正在使用 Real-ESRGAN 对视频进行逐帧超分辨率。速度极慢！")
                
            out_width = int(width * upscale_factor)
            out_height = int(height * upscale_factor)
            
            # 更新 real_strength 的基准
            real_strength = shift_strength * (out_width / target_scale_width)
        else:
            real_strength = shift_strength * (width / target_scale_width)
            
        # 优先使用 avc1 (H.264) 编码，兼容性最好。如果失败回退到 mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            if not out.isOpened():
                print("提示: avc1 编码器不可用，尝试使用 mp4v 编码...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        except Exception as e:
            print(f"编码器初始化异常: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        if not out.isOpened():
            print("错误: 无法创建视频文件。可能是路径问题或缺少编码器。尝试安装 opencv-python-headless 或检查输出路径。")
            cap.release()
            return
        
        pbar = tqdm(total=total_frames, unit="frame")
        
        border_x = int(real_strength) + int(out_width * 0.02)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 0. AI 增强 
            if converter.enhancer:
               try:
                   if hasattr(converter.enhancer, 'model_name') and converter.enhancer.model_name == 'fsrcnn':
                       frame = converter.enhancer.enhance(frame)
                   else:
                       frame = converter.enhancer.enhance(frame, out_scale=upscale_factor)
               except Exception as e:
                   print(f"Frame enhance error: {e}")
                
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
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--strength", type=float, default=25.0, help="立体分离强度 (默认 25)")
    parser.add_argument("--model", default="DPT_Large", choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], help="深度模型类型")
    
    # 画质增强选项
    parser.add_argument("--enhance", action="store_true", help="启用画质增强 (默认使用快速模式 FSRCNN)")
    parser.add_argument("--hq", action="store_true", help="启用高质量最慢模式 (Real-ESRGAN)")
    parser.add_argument("--scale", type=float, default=2.0, help="画质增强缩放倍数 (仅 HQ 模式有效，Fast 模式固定 x2)")
    
    args = parser.parse_args()
    
    # Determine enhance mode
    enhance_mode = None
    if args.hq:
        enhance_mode = "hq"
    elif args.enhance:
        enhance_mode = "fast"
    
    if not os.path.exists(args.input):
        print(f"未找到输入文件: {args.input}")
        exit(1)
        
    output = args.output
    if not output:
        name, ext = os.path.splitext(args.input)
        tag = "_hq_3d" if args.hq else ("_enhanced_3d" if args.enhance else "_3d")
        output = f"{name}{tag}{ext}"
        
    converter = Converter2Dto3D(model_type=args.model, enhance_mode=enhance_mode)
    process_file(args.input, output, converter, shift_strength=args.strength, upscale_factor=args.scale)
