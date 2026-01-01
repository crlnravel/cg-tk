# Sketch-to-Material Studio

A computer graphics application that combines classical OpenGL rendering with AI-based texture generation. Transform your 2D sketches into interactive 3D materials using Stable Diffusion and depth estimation models.

![Result](assets/result.png)

## Demo Video

Watch the demo videos to see the application in action:

📹 [Demo Video](https://drive.google.com/file/d/1rqnt2-XbmlUApWPZ-2AUKFLzE8TArEf4/view?usp=sharing)

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

1. **Draw Sketch**
   - Draw on the canvas with your mouse (white brush on black background)
   - Use **[** and **]** keys to adjust brush size
   - Click "Clear Canvas" button to reset the canvas
   - Use template buttons (numbered) to load example sketches

   ![Input Sketch](assets/sketch_processed.png)

2. **Generate 3D Material**
   - Enter a project name and text prompt in the sidebar
   - Click **GENERATE** button to start generation
   - Wait for AI processing (may take 1-3 minutes depending on hardware)

3. **View 3D Result**
   - The application automatically switches to 3D view mode
   - Use keyboard controls to navigate (see Controls section)
   - Press **ESC** to return to paint mode

   ![3D Result](assets/result.png)

### Controls

#### Paint Mode
- **Mouse Drag**: Draw on canvas (white brush)
- **[** / **]** Keys: Decrease / Increase brush size
- **Clear Canvas Button**: Reset the canvas
- **Template Buttons**: Load example sketches
- **Load Project Button**: Load previously generated materials

#### 3D View Mode
- **W / S**: Rotate camera up / down
- **A / D**: Rotate camera left / right
- **R / E**: Zoom in / out
- **H**: Toggle help overlay
- **ESC**: Return to paint mode

### Generated Texture Maps

The application generates several texture maps for realistic 3D rendering:

| Albedo | Depth | Normal | Roughness |
|--------|-------|--------|-----------|
| ![Albedo](assets/albedo.png) | ![Depth](assets/depth.png) | ![Normal](assets/normal.png) | ![Roughness](assets/roughness.png) |

## Authors

- **Carleano Ravelza Wongso** - carleano.ravelza@ui.ac.id
- **Andrew Devito Aryo** - andrew.devito@ui.ac.id
- **Arya Raditya Kusuma** - arya.raditya@ui.ac.id
- **Tristan Agra Yudhistira** - tristan.agra@ui.ac.id

---

**CSCE604029 • Computer Graphics • Semester Gasal 2025/2026**  
**Fakultas Ilmu Komputer, Universitas Indonesia**