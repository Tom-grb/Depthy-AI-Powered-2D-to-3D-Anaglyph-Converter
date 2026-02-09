# 2D to 3D Converter (Red-Cyan Anaglyph)

This tool converts standard 2D images and videos into Red-Cyan 3D anaglyph format using AI depth estimation (MiDaS).

## Requirements

1.  Python 3.8+
2.  **Setup Virtual Environment (Recommended):**
    ```bash
    # Create virtual environment
    python -m venv venv
    
    # Activate (Windows)
    .\venv\Scripts\activate
    
    # Activate (Linux/Mac)
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Note: You should have PyTorch installed with CUDA support if you have a GPU. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for specific install commands if needed. Typically: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` (or similar).

## Usage

### Process an Image
```bash
python main.py path/to/image.jpg
```

### Process a Video
```bash
python main.py path/to/video.mp4
```

### Options
*   `--strength`: Adjust the 3D effect strength (Depth). Default is 25. Higher means more "pop out" but risk of eye strain.
    ```bash
    python main.py image.jpg --strength 40
    ```
*   `--model`: Choose the depth model.
    *   `DPT_Large` (Default): Best quality, slower.
    *   `DPT_Hybrid`: Good balance.
    *   `MiDaS_small`: Fastest, lower quality.
    ```bash
    python main.py video.mp4 --model MiDaS_small
    ```

## Output
The output file will be saved in the same folder with `_3d` appended to the name (e.g., `image_3d.jpg`).
