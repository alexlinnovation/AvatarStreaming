# Ditto: Custom Talking Head Synthesis

<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
</div>

A full-stack application for real-time talking head synthesis with a Next.js frontend and Python backend powered by TensorRT inference.

## 📁 Project Structure

frontend/                 # Next.js frontend application
├── src/                 # Source code
├── pages/               # Next.js pages
├── styles/              # Styling files
├── transcriptions/      # Transcription files
├── env/                 # Environment configurations
├── package.json         # Frontend dependencies
└── ...                  

src/                     # Python backend
├── __pvcache__/         # Cache directory
├── static/              # Static files
├── tests/               # Test files
├── uploads/             # Upload directory
├── app.py               # Main application
├── inference.py         # Inference logic
├── livekit_video.py     # LiveKit integration
├── livekit_video_keepalive.py  # Keepalive service
└── ...

scripts/                 # Utility scripts
├── gated_avsync.py      # Audio-video sync
├── gated_avsyncv2.py    # Enhanced AV sync
├── qwen.py              # Qwen model integration
└── utils.py             # Utility functions

.env.example            # Environment template
.eslintrc.json          # ESLint configuration
.gitignore              # Git ignore rules
Dockerfile              # Docker configuration
package.json            # Project dependencies
README.md               # This file
requirements.txt        # Python dependencies
tsconfig.json           # TypeScript config

## 🛠️ Installation

Tested Environment  
- System: Ubuntu
- GPU: RTX 3090  
- Python: 3.10  
- tensorRT: 8.6.1  

## 📥 Docker
Check the
```bash
Dockerfile
```

## 🛠️ Installation

### System Requirements
- Base Image: nvcr.io/nvidia/tensorrt:23.12-py3
- System: Ubuntu
- GPU: RTX 3090 (or compatible)
- Python: 3.10
- TensorRT: 8.6.1

### System Dependencies
- build-essential
- python3-dev python3.10-dev
- ffmpeg
- libsndfile1
- libturbojpeg0-dev
- libsm6 libxext6 libgl1
- git curl wget

### Python Dependencies
- numpy==2.0.1
- kokoro-onnx==0.3.9
- onnxruntime-gpu==1.22.0
- Other requirements from requirements.txt

### Installation Methods

#### Docker Setup
Check the Dockerfile for containerized deployment.

#### Conda Environment
Create conda environment:
conda env create -f environment.yaml
conda activate ditto

#### Pip Installation
pip install \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1


## 🚀 Running the Application

### Backend Server
Start the Python backend:
uvicorn livekit_video_keepalive:app --host 0.0.0.0 --port 8010 --reload

### Frontend Development
Start the Next.js frontend:
cd frontend
npm install
npm run dev

## 📚 Citation
If you find this codebase useful for your research, please use the following entry.
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}

### Conda
Create `conda` environment:
```bash
conda env create -f environment.yaml
conda activate ditto
```

### Pip
```bash
pip install \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1
```

## 📥 Download Checkpoints
Download checkpoints from [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead) and put them in `checkpoints` dir:
```bash
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

The `checkpoints` should be like:
```text
./checkpoints/
├── ditto_cfg
│   ├── v0.4_hubert_cfg_trt.pkl
│   └── v0.4_hubert_cfg_trt_online.pkl
├── ditto_onnx
│   ├── appearance_extractor.onnx
│   ├── blaze_face.onnx
│   ├── decoder.onnx
│   ├── face_mesh.onnx
│   ├── hubert.onnx
│   ├── insightface_det.onnx
│   ├── landmark106.onnx
│   ├── landmark203.onnx
│   ├── libgrid_sample_3d_plugin.so
│   ├── lmdm_v0.4_hubert.onnx
│   ├── motion_extractor.onnx
│   ├── stitch_network.onnx
│   └── warp_network.onnx
└── ditto_trt_Ampere_Plus
    ├── appearance_extractor_fp16.engine
    ├── blaze_face_fp16.engine
    ├── decoder_fp16.engine
    ├── face_mesh_fp16.engine
    ├── hubert_fp32.engine
    ├── insightface_det_fp16.engine
    ├── landmark106_fp16.engine
    ├── landmark203_fp16.engine
    ├── lmdm_v0.4_hubert_fp32.engine
    ├── motion_extractor_fp32.engine
    ├── stitch_network_fp16.engine
    └── warp_network_fp16.engine
```

- The `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` is online config
- The `ditto_cfg/v0.4_hubert_cfg_trt.pkl` is offline config


## 📚 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}
```
