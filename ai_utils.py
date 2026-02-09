import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import cv2
import numpy as np

# --- 1. Model Architecture (RRDBNet) ---
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB, nb, nf=nf, gc=gc)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

# --- 2. Enhancer Class ---

class ImageEnhancer:
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = None
        
    def load_model(self):
        if self.model is not None:
            return
            
        print("正在初始化画质增强模型 (Real-ESRGAN)...")
        # Initialize model
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
        
        # Download weights if needed
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_path = os.path.join(os.path.expanduser("~"), ".cache", "RealESRGAN_x4plus.pth")
        
        if not os.path.exists(model_path):
            print(f"下载增强模型权重中: {model_url}")
            torch.hub.download_url_to_file(model_url, model_path)
            
        # Load weights
        loadnet = torch.load(model_path, map_location=self.device)
        # Handle 'params_ema' key if present (RealESRGAN structure)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
            
        self.model.load_state_dict(loadnet[keyname], strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("画质增强模型加载完成。")

    def enhance(self, img, out_scale=2):
        """
        img: numpy array (H, W, 3) BGR (OpenCV format)
        out_scale: Output scaling factor (standard model is 4x internally, we can resize result)
        Returns: numpy array (H_new, W_new, 3) BGR
        """
        self.load_model()
        
        # Determine tile size based on GPU memory
        # For a 3050 Laptop (4GB), conservative tile size is good.
        tile_size = 200 # Process 200x200 patches
        tile_pad = 10
        
        # Pre-process
        img = img.astype(np.float32) / 255.
        img_tensor = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Tile processing
        b, c, h, w = img_tensor.shape
        output_height = h * 4
        output_width = w * 4
        output_shape = (b, c, output_height, output_width)
        
        # Start tiling
        output = torch.zeros(output_shape, device=self.device, dtype=torch.float32)
        
        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)
        
        # Iterate tiles
        # print(f"Enhancing image ({w}x{h}) in {tiles_x}x{tiles_y} tiles...")
        
        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Extract tile coordinates
                    ofs_x = x * tile_size
                    ofs_y = y * tile_size
                    
                    # Input tile range
                    start_x = ofs_x
                    end_x = min(ofs_x + tile_size, w)
                    start_y = ofs_y
                    end_y = min(ofs_y + tile_size, h)
                    
                    # Padded tile range (for context)
                    start_x_pad = max(start_x - tile_pad, 0)
                    end_x_pad = min(end_x + tile_pad, w)
                    start_y_pad = max(start_y - tile_pad, 0)
                    end_y_pad = min(end_y + tile_pad, h)
                    
                    # Crop input
                    input_tile = img_tensor[:, :, start_y_pad:end_y_pad, start_x_pad:end_x_pad]
                    
                    # Run model
                    try:
                        output_tile = self.model(input_tile)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                             # Fallback or very clear error
                             print("显存不足 (OOM)，请尝试减小 tile_size。")
                             torch.cuda.empty_cache()
                        raise e

                    # Output tile range (mapped to 4x)
                    output_start_x = start_x * 4
                    output_end_x = end_x * 4
                    output_start_y = start_y * 4
                    output_end_y = end_y * 4
                    
                    # Padded output range
                    output_start_x_tile = (start_x - start_x_pad) * 4
                    output_end_x_tile = output_start_x_tile + (end_x - start_x) * 4
                    output_start_y_tile = (start_y - start_y_pad) * 4
                    output_end_y_tile = output_start_y_tile + (end_y - start_y) * 4
                    
                    # Place into output
                    output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                        output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        
        # Post-process
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # RGB -> BGR
        output = (output * 255.0).round().astype(np.uint8)
        
        # Resize to desired out_scale (Model is fixed 4x)
        if out_scale != 4:
            new_h = int(h * out_scale)
            new_w = int(w * out_scale)
            output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
        return output
