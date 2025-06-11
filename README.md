<h2 align='center'>Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis</h2>

<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
</div>


## 🛠️ Installation

Tested Environment  
- System: Centos 7.2  
- GPU: A100  
- Python: 3.10  
- tensorRT: 8.6.1  


Clone the codes from [GitHub](https://github.com/antgroup/ditto-talkinghead):  
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead
```

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
