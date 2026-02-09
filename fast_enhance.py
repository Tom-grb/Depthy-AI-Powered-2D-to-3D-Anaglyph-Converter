import cv2
import os

class FastEnhancer:
    def __init__(self, use_gpu=True):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model_name = "fsrcnn" # fast super resolution cnn
        self.scale = 2
        # FSRCNN is extremely fast and provides decent sharpening/upscaling compared to bicubic
        # Alternative: ESPCN
        
        # Download heavy weights only if needed?
        # OpenCV's models need .pb files.
        # Let's check or download logic here.
        self.model_path = os.path.join(os.path.expanduser("~"), ".cache", "FSRCNN_x2.pb")
        self.use_gpu = use_gpu

    def load_model(self):
        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
        if not os.path.exists(self.model_path):
            print(f"下载 FSRCNN 轻量模型: {url}")
            import urllib.request
            try:
                urllib.request.urlretrieve(url, self.model_path)
            except Exception as e:
                print(f"下载模型失败，请手动下载 FSRCNN_x2.pb 到 {self.model_path}")
                raise e
        
        self.sr.readModel(self.model_path)
        self.sr.setModel("fsrcnn", 2)
        
        if self.use_gpu:
            # Try to set CUDA/OpenCL backend if available in OpenCV
            # Standard pip opencv-python usually doesn't include CUDA support
            # but we can try OpenCL which is often default or setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # Actually, standard opencv-python is CPU only for DNN usually unless compiled manually.
            # But FSRCNN is so small it runs fast on CPU too.
            try:
                self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except:
                pass

    def enhance(self, img):
        # Input: BGR numpy array
        # Output: BGR numpy array 2x size
        if not os.path.exists(self.model_path):
             self.load_model()
             
        result = self.sr.upsample(img)
        return result

    def sharpen(self, img):
        # Conventional unsharp mask for speed
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
