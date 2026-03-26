# Usage

## Environment
Requires ROS2 Humble. The node subscribes to stereo image topics, runs inference, and publishes depth + point cloud.

## Weights

Download from [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and place under `weights/`.

| Checkpoint | PyTorch (ms) | Python-TRT (ms) | Notes |
|-----------|-------------|---------|-------|
| `23-36-37` | 49.4 | 23.4 | Best accuracy |
| `20-26-39` | 43.6 | 19.4 | Balanced |
| `20-30-48` | 38.4 | 16.6 | Fastest |

> Profiled on RTX 3090 Ti, 480×640, `valid_iters=8`.


| Checkpoint | FPS (Hz) | CPP-TRT (ms) | Notes |
|-----------|-------------|---------|-------|
| `23-36-37` | TBD | TBD | Best accuracy |
| `20-26-39` | 12.8 | 77.8 | Balanced |
| `20-30-48` | TBD | TBD | Fastest |

> Profiled on Jetson Orin AGX, 480×640, `valid_iters=4`.



## TensorRT (Tested on 10.3.0/10.7.0)

```bash
# Export ONNX
python scripts/make_onnx.py \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/23-36-37/ --height 480 --width 640

# Build engine (via trtexec)
/usr/src/tensorrt/bin/trtexec --onnx=output/23-36-37/feature_runner.onnx --saveEngine=output/23-36-37/feature_runner_fp16.engine --fp16  --useCudaGraph
/usr/src/tensorrt/bin/trtexec --onnx=output/23-36-37/post_runner.onnx --saveEngine=output/23-36-37/post_runner_fp16.engine --fp16  --useCudaGraph

```

## Demo

```bash
# Inference Once
./install/fast_foundation_stereo/lib/fast_foundation_stereo/fast_foundation_stereo_test

# ROS Node/Launch
ros2 launch ros_dnn_stereo dnn_stereo_depth.launch.py
ros2 bag play xxx.bag

```


