# Sketch-to-Material Studio

A computer graphics application that combines classical OpenGL rendering with AI-based texture generation. Transform your 2D sketches into interactive 3D materials using Stable Diffusion and depth estimation models.

![Result](readme_images/result.jpeg)

## Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended) or Apple Silicon (MPS) or CPU (slower)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: ~10GB free space for model downloads

## Installation

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv env
env\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Pre-Download AI Models (Recommended)

Run this script once to download all AI models (~4-5GB):

```bash
python prepare_models.py
```

**Note**: If you skip this step, models will be downloaded automatically on first run.

## Usage

### Running the Application

```bash
python main_sketch.py
```

### Basic Workflow

1. **Draw or Upload Sketch**
   - Draw on the canvas with your mouse
   - Or drag & drop an image file onto the window
   - Use number keys (1-5) to select colors
   - Press 'C' to clear the canvas

   ![Input Sketch](readme_images/sketch_processed.png)

2. **Generate 3D Material**
   - Press **ENTER** to start generation
   - Enter a text prompt in the console (e.g., "wood texture", "brick wall", "marble")
   - Wait for AI processing (may take 1-3 minutes depending on hardware)

3. **View 3D Result**
   - The application automatically switches to 3D view mode
   - Drag with left mouse button to rotate camera
   - Press **ESC** to return to paint mode

   ![3D Result](readme_images/result.jpeg)

### Generated Texture Maps

The application generates several texture maps for realistic 3D rendering:

| Albedo | Depth | Normal | Roughness |
|--------|-------|--------|-----------|
| ![Albedo](readme_images/albedo.png) | ![Depth](readme_images/depth.png) | ![Normal](readme_images/normal.png) | ![Roughness](readme_images/roughness.png) |

### Controls

- **Mouse Drag**: Draw on canvas (paint mode) / Rotate camera (3D mode)
- **1-5 Keys**: Select color from palette
- **C Key**: Clear canvas
- **ENTER**: Generate 3D material from sketch
- **ESC**: Return to paint mode from 3D view
- **Drag & Drop**: Load image file onto canvas
