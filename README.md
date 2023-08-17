# DSVT-AI-TRT
DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets(CVPR2023),vaymo vehicle 3D Object Detection(top2), waymo cyclist 3D Object Detection(top1),waymo pedestrian 3D Object Detection(top1)

DSVT-AI-TRT(**DSVT   ALL IN TensorRT**,NMS not implemented in TensorRT,implemented in c++) 

DSVT  consists of six parts:
- preprocess: generate pillars, it is implemented in ./plugins/src/points2Features.cu,it is a TensorRT plugin
- 3D backbone: 3D backbone include input _dsvt_layer and dsvt block.    ./plugins/src/getSet.cu is the core TensorRT plugin for input_dsvt_layer,  multiheadattention is  main one for dsvt block,it mainy implemented by TensorRT AIP, gelu and layernorm is implemented by  TensorRT plugin .you can find them in ./plugins/gelu.cu and ./plugins/layernorm.cu.
- 2D backbon: 2D bev resnet backbone, implemented by TensorRT AIP
- head: similar to centerpoint head,, this part is mainy implemented by TensorRT aip.
- postprocess: filter_box_by_score is implemented by filterBoxByScore.cu, it is also a plugin.
- 3D NMS: it comes from  https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/src/postprocess.cpp

- for details, please refer to ./tools/dsvt_cbgs_dyn_pp_centerpoint.yaml


## Config

- all config in params.h
- FP16/FP32 can be selected by USE_FP16 in params.h
- GPU id can be selected by DEVICE in params.h
- NMS thresh can be modified by NMS_THRESH in params.h

## How to Run

1. **build DSVT-AI-TRT and run**

```
firstly, install TensorRT,my environment is ubuntu 18.04, nvidia-driver 470.94  cuda 10.2,cudnn8.2.
I installed TensorRT with TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz.

after that, modify CMakeLists.txt
include_directories(/home/xxx/softwares/nvidia/TensorRT-8.2.1.8/include)
link_directories(/home/xxx/softwares/nvidia/TensorRT-8.2.1.8/lib)
Change these two lines to your own path

cd DSVT-AI-TRT
mkdir build
cd build
cmake ..
make
sudo ./dsvt-ai-trt -s             // serialize model to plan file i.e. 'dsvt-ai-trt.engine'
sudo ./dsvt-ai-trt -d    // deserialize plan file and run inference, lidar points will be processed.
predicted outputs saved in dsvt-AI-TRT/data/outputs folder

```
**one frame takes about 0.7 seconds on my laptop with Intel(R) Core(TM) i5-7300HQ and NVIDIA GeForce GTX 1050 Mobile(1050ti)**

2. **show predicted 3D boxes in the lidar frame** 

```
fristly install anaconda.
then,
1 conda create -n pc_show python=3.6
2 conda activate pc_show
3 pip install vtk==8.1.2
4 pip install mayavi
5 pip install PyQt5
6 pip install opencv-python

cd tools
python show_box_in_points.py

if An error occurred like: qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/xxx/anaconda3/envs/pc_show/lib/python3.6/site-packages/cv2/qt/plugins" even though it was found.
just delete it by run: sudo rm -rf /home/xxx/anaconda3/envs/pc_show/lib/python3.6/site-packages/cv2/qt/plugins
and try again

warning: do not close current Mayavi Scene window, type c in running terminal and press Enter,
it will show next lidar frame with predited 3d boxes in current Mayavi Scene window. 

```
![Image text](https://raw.githubusercontent.com/jingyue202205/SE-SSD-AI-TRT/master/pics/000010.png)

3. **generate wts** 

```
1 you can copy  dsvt_cbgs_dyn_pp_centerpoint.yaml and nuscenes_dataset.yaml to [DSVT](https://github.com/Haiyang-W/DSVT) , retrain model.
in nuscenes_dataset.yaml, I used unscenes v1.0-mini dataset, to save training time.  only one point cloud frame was used.
2 refer to ./tools/gen_wts.py, generate new wts file.

```
## More Information

Reference code:

[DSVT](https://github.com/Haiyang-W/DSVT)  

[tensorrtx](https://github.com/wang-xinyu/tensorrtx) 

[tensorrt_plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin)

[CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)

[frustum_pointnets_pytorch](https://github.com/simon3dv/frustum_pointnets_pytorch)





