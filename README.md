# smart-parking

Parking/Security system for monitoring cars
-------------------------------------------
Run an optimized "yolov4-416-tiny" object detector at ~24 FPS on Jetson Nano.


Table of contents
-----------------
* [Acknowlegment](#acknowlegment)    
* [Prerequisite](#prerequisite)
* [Demo: YOLOv4](#yolov4)
* [Demo: Running the App](#app)

<a name="acknowlegment"></a>
Acknowlegment
------------

 This is an open source project base on [The AI Guy's](https://github.com/theAIGuysCode) custom Yolov4 model [repo](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial) and [JK Jung's](https://github.com/jkjung-avt) tensorrt demos [repo](https://github.com/jkjung-avt/tensorrt_demos). The following instructions are a summary from tensorrt demos to build and run an optimized version of Yolov4. I trained a custom version of Yolov4 tiny to detect two objects, license plates and cars. The weights and config file are already located inside this repo, so their is no need to train your own model. In case you want to train you own model, here is a [tutorial](https://www.youtube.com/watch?v=mmj3nxGT2YQ&t=434s&ab_channel=TheAIGuy) for training custom versions of Yolov4 in the cloud for free! Special thanks to [The AI Guy](https://github.com/theAIGuysCode) and [JK Jung](https://github.com/jkjung-avt) for sharing their knowledge.

<a name="prerequisite"></a>
Prerequisite
------------

  The first step is to install a JetPack copy on your Jetson system. If you have not done this, follow the [Nvidia tutorial](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit). In my case, I have JetPack-4.6. Also JK Jung recomeds following these steps, [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/) and/or [Setting up Jetson Xavier NX](https://jkjung-avt.github.io/setting-up-xavier-nx/).

The target Jetson system must have TensorRT libraries installed.

* Demo: requires TensorRT 6.x+.
* Demo: INT8 requires TensorRT 6.x+ and only works on GPUs with CUDA compute 6.1+.


You could check which version of TensorRT has been installed on your Jetson system. Run the following commnad in the terminal.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
```
I got this on my JetPack version 4.6:

```shell
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.8
/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.0.1
```

This demo uses the opencv version included with JetPack (4.1.1).If you'd prefer building your own, refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) for how to build from source and install opencv-3.4.6 on your Jetson system.


You need to have "protobuf" installed for this demo.  Mr. Jung recommends installing "protobuf-3.8.0" using [install_protobuf-3.8.0.sh](https://github.com/jkjung-avt/jetson_nano/blob/master/install_protobuf-3.8.0.sh) script.  This script would take a couple of hours on a Jetson system.  Alternatively, pip3 install a recent version of "protobuf" should also work (but might run a little bit slowlier).

In case you are setting up a Jetson Nano or Jetson Xavier NX from scratch check this first.

* [JetPack-4.5](https://jkjung-avt.github.io/jetpack-4.5/)

<a name="yolov4"></a>
YOLOv4
---------------

Convert pre-trained yolov3 and yolov4 models through ONNX to TensorRT engines.

1. Install "pycuda".

   ```shell
   $ cd ${HOME}/project/smart-parking
   $ ./install_pycuda.sh
   ```

2. Install **version "1.4.1" (not the latest version)** of python3 **"onnx"** module.  Note that the "onnx" module would depend on "protobuf" as stated in the [Prerequisite](#prerequisite) section.  Reference: [information provided by NVIDIA](https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/post/5347666/#5347666).

   ```shell
   $ sudo pip3 install onnx==1.4.1
   ```

3. Go to the "plugins/" subdirectory and build the "yolo_layer" plugin.  When done, a "libyolo_layer.so" would be generated.

   ```shell
   $ cd ${HOME}/project/smart-parking/plugins
   $ make
   ```

4. Convert the targeted model to ONNX and then to TensorRT engine.  I used "yolov4-tiny-custom" but if you want to try your own just verify that the weights file and config have the same name, example: “yolov4-tiny-custom.cfg” and “yolov4-tiny-custom.weights”. You may skip this step as it is already included in the directory. 

   ```shell
   $ cd ${HOME}/project/smart-parking/yolo
   $ python3 yolo_to_onnx.py -m yolov4-tiny-custom
   $ python3 onnx_to_tensorrt.py -m yolov4-tiny-custom
   ```

   The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit.  When that is done, the optimized TensorRT engine would be saved as "yolov4-tiny-custom.trt".

   In case "onnx_to_tensorrt.py" fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine.  This problem might be solved by adding a larger swap file to the system.  Reference: [Process killed in onnx_to_tensorrt.py Demo#5](https://github.com/jkjung-avt/tensorrt_demos/issues/344).


5. Check out JK Jung's blog posts for implementation details:

   * [TensorRT ONNX YOLOv3](https://jkjung-avt.github.io/tensorrt-yolov3/)
   * [TensorRT YOLOv4](https://jkjung-avt.github.io/tensorrt-yolov4/)
   * [Verifying mAP of TensorRT Optimized SSD and YOLOv3 Models](https://jkjung-avt.github.io/trt-detection-map/)
   * For training your own custom yolov4 model: [Custom YOLOv4 Model on Google Colab](https://jkjung-avt.github.io/colab-yolov4/)
   * For adapting the code to your own custom trained yolov3/yolov4 models: [TensorRT YOLO For Custom Trained Models (Updated)](https://jkjung-avt.github.io/trt-yolo-custom-updated/)


<a name="app"></a>
Running the App
---------------

Lets install some dependencies related to the script and and add your API key to cfg.json. 

1. Install Shapely.

   ```shell
   $ cd ${HOME}/project/smart-parking
   $ ./install_shapely.sh
   ```

2. In order to detect the car's license plate number we will use an OCR Api called Plate Recognizer. The advantages of using this api are that it was specifically trained for plate recognition (robust model) and it gives you 2500 calls per month for free. Create your account at [Plate Recognizer](https://platerecognizer.com/) and add you api token to cfg.json.

   * "KEY": "************************************************"

3. Run inference over video.

   ```shell
      $ cd ${HOME}/project/smart-parking
      $ python3 trt_yolo_cv.py -v videos/my_lp_10.mp4 -m yolov4-tiny-custom -s -d -z
   ```
   View Results:

   [![Car detected](https://github.com/Angel-Ceballos/smart-parking/blob/6706f88e14b9b01a7d699ca911cf6ffa2b14eeb7/docs/car_detection.png?raw=true)](https://youtu.be/fSZR6RCJQnk)

   Flag Description
   * `--video videos/my_lp_10.mp4`: a video to run the inference.
   * `--output detections/demo_lp_10.mp4`: file name to save the prediction 
   * `--show_vid`: display results live.
   * `--detection_zone`: draws detection zone.
   * `--detect_car`: enables car detection and plate recognizer.
   * `--threshold 0.5`: sets threshold for object detection.
   * `--model yolov4-tiny-custom`: loads model.

Licenses
--------

1. I referenced source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop most of the demos in this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).
4. [TensorFlow Object Detection Models](https://github.com/tensorflow/models/tree/master/research/object_detection): [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
5. YOLOv3/YOLOv4 models ([DarkNet](https://github.com/AlexeyAB/darknet)): [YOLO LICENSE](https://github.com/AlexeyAB/darknet/blob/master/LICENSE).
7. For the rest of the code (developed by jkjung-avt and other contributors): [MIT License](https://github.com/jkjung-avt/tensorrt_demos/blob/master/LICENSE).

