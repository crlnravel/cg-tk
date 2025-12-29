# Sketch-to-Material Studio

A computer graphics application that combines classical OpenGL rendering with AI-based texture generation. Transform your 2D sketches into interactive 3D materials using Stable Diffusion and depth estimation models.

## 🎯 Project Overview

This project demonstrates the integration of:
- **Classical Computer Graphics**: OpenGL rasterization pipeline, Phong lighting model, texture mapping, displacement mapping
- **AI-based Graphics**: Stable Diffusion Img2Img for texture synthesis, learned depth estimation, and normal map generation

## ✨ Features

- **Interactive 2D Canvas**: Draw sketches with mouse, drag & drop images, color palette selection
- **AI Texture Generation**: Convert sketches to detailed textures using Stable Diffusion
- **Automatic Geometry Extraction**: Generate depth and normal maps from generated textures
- **Real-time 3D Rendering**: View your materials in 3D with interactive camera controls
- **Classical Lighting**: Phong illumination model with ambient, diffuse, and specular components

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended) or Apple Silicon (MPS) or CPU (slower)
- **RAM**: Minimum 8GB (16GB+ recommended for AI models)
- **OS**: Windows, macOS, or Linux

### Hardware Recommendations
- **GPU**: NVIDIA GPU with 6GB+ VRAM for best performance
- **CPU**: Multi-core processor for depth estimation
- **Storage**: ~10GB free space for model downloads

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd cg-tk
```

### Step 2: Create Virtual Environment (Recommended)

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

### Step 3: Install Dependencies

**For CUDA (NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**For Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎮 Usage

### Running the Application

```bash
python main_sketch.py
```

**Note**: On first run, the application will download AI models (~4-5GB). This may take several minutes depending on your internet connection.

### Workflow

1. **Draw or Upload Sketch**
   - Draw directly on the canvas with your mouse
   - Or drag & drop an image file onto the window
   - Use number keys (1-5) to select colors from the palette
   - Press 'C' to clear the canvas

2. **Generate 3D Material**
   - Press **ENTER** to start generation
   - Enter a text prompt in the console (e.g., "wood texture", "brick wall", "marble")
   - Wait for AI processing (may take 1-3 minutes depending on hardware)

3. **View 3D Result**
   - The application automatically switches to 3D view mode
   - Drag with left mouse button to rotate camera
   - Press **ESC** to return to paint mode

### Controls

#### Paint Mode
- **Mouse Drag**: Draw on canvas
- **1-5 Keys**: Select color from palette
  - `1`: Black
  - `2`: White
  - `3`: Red
  - `4`: Green
  - `5`: Blue
- **C Key**: Clear canvas
- **ENTER**: Generate 3D material from sketch
- **Drag & Drop**: Load image file onto canvas

#### 3D View Mode
- **Left Mouse Drag**: Rotate camera around object
- **ESC**: Return to paint mode

## 📁 Project Structure

```
cg-tk/
├── main_sketch.py          # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── sketch_input.png       # Saved sketch (generated)
├── assets_albedo.png      # Generated albedo texture (generated)
├── assets_depth.png        # Generated depth map (generated)
├── assets_normal.png      # Generated normal map (generated)
└── env/                   # Virtual environment (not included)
```

## 🔧 Technical Details

### Classical Graphics Components
- **Rasterization Pipeline**: OpenGL 3.3 Core Profile
- **Lighting Model**: Phong (ambient + diffuse + specular)
- **Texture Mapping**: Multi-texture (albedo, depth, normal)
- **Geometry**: 100x100 grid mesh with displacement mapping
- **Shaders**: Custom GLSL vertex and fragment shaders

### AI Components
- **Texture Generation**: Stable Diffusion v1.5 (Img2Img)
- **Depth Estimation**: Depth-Anything-Small model
- **Normal Map**: Generated from depth using Sobel filters

### Performance Notes
- **First Run**: Model downloads take 5-10 minutes
- **Generation Time**: 
  - GPU (CUDA): ~30-60 seconds
  - Apple Silicon (MPS): ~60-120 seconds
  - CPU: ~5-10 minutes
- **Rendering**: 60 FPS on modern hardware

## 🐛 Troubleshooting

### Common Issues

**1. "CUDA out of memory" error**
- **Solution**: The models require significant VRAM. Try:
  - Close other GPU-intensive applications
  - Reduce batch size in code (if applicable)
  - Use CPU mode (slower but works)

**2. Models not downloading**
- **Solution**: 
  - Check internet connection
  - Models are downloaded from Hugging Face Hub
  - First download may take 10-15 minutes
  - Ensure you have ~5GB free space

**3. OpenGL errors on macOS**
- **Solution**: The code includes macOS-specific fixes. If issues persist:
  - Update macOS to latest version
  - Ensure you're using Python 3.8+

**4. Pygame window not responding**
- **Solution**: 
  - During AI generation, the window may freeze (blocking operation)
  - This is normal - wait for console output
  - Consider adding threading for non-blocking generation (future improvement)

**5. Import errors**
- **Solution**: 
  - Ensure virtual environment is activated
  - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
  - Check Python version: `python --version` (should be 3.8+)

**6. Slow performance on CPU**
- **Solution**: 
  - CPU mode is significantly slower (5-10 minutes per generation)
  - Consider using Google Colab with GPU for faster results
  - Or use a machine with NVIDIA GPU

### Platform-Specific Notes

**Windows:**
- Ensure you have Visual C++ Redistributables installed
- PyOpenGL may require additional system libraries

**macOS:**
- OpenGL context is automatically configured for macOS compatibility
- MPS (Metal Performance Shaders) support is included

**Linux:**
- May require: `sudo apt-get install python3-opengl`
- Ensure graphics drivers are up to date

## 📊 Output Files

The application generates the following files:

- `sketch_input.png`: Your input sketch (saved when generating)
- `assets_albedo.png`: Generated texture/albedo map (512x512)
- `assets_depth.png`: Depth map for displacement (512x512)
- `assets_normal.png`: Normal map for surface details (512x512)

## 🎓 Academic Context

This project fulfills requirements for:
- **Classical Graphics**: Rasterization, lighting models, texture mapping
- **AI Graphics**: Diffusion models, learned depth estimation, generative textures
- **Integration**: Seamless combination of classical and AI techniques

## 📝 Notes

- **Model Size**: First run downloads ~4-5GB of AI models
- **Internet Required**: For initial model download only
- **GPU Recommended**: For reasonable generation times
- **Blocking Operations**: AI generation blocks the UI (future improvement: threading)

## 🔗 References

- Stable Diffusion: https://huggingface.co/runwayml/stable-diffusion-v1-5
- Depth-Anything: https://huggingface.co/LiheYoung/depth-anything-small-hf
- PyGame: https://www.pygame.org/
- PyOpenGL: http://pyopengl.sourceforge.net/

## 👥 Authors

[Add your group members and NPM here]

## 📄 License

[Add your license information here]

---

**Happy Creating! 🎨**

For issues or questions, please refer to the troubleshooting section or check the code comments.

