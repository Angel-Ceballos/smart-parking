"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse
from pickle import TRUE
import time
import datetime
from pprint import pprint
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.functions import *
from classes.car import Car

def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str,
        help='output video file name')
    parser.add_argument(
        '-s', '--show_vid', action='store_true',
        help='displays video with bboxes')
    parser.add_argument(
        '-z', '--detection_zone', action='store_true',
        help='shows detection zone and track')
    parser.add_argument(
        '-d', '--detect_car', action='store_true',
        help='enables car detection')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.3,
        help='threshold for inference')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo, conf_th, vis, writer, args, video_name):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    curr_frame = 0
    cropped_frames = 0
    crop_num = 3

    show = args[0]
    write = args[1]
    draw_zone = args[2]
    detect = args[3]
    car_list = []

    
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        if frame is None:   break
        clean_frame = frame.copy()
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        frame, is_valid, valid_list = vis.draw_bboxes(frame, boxes, confs, clss, draw_zone, detect)
        
        if detect and is_valid and cropped_frames < crop_num:
            print(f"Cropping frame: {curr_frame}")
            if cropped_frames == 0:
                crop_path = os.path.join(os.getcwd(), "detections", "crop", video_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
        
            if draw_zone:
                h, _, _ = frame.shape
                height_ration = int(h / 25)
                frame = cv2.putText(frame, f"Cropping frame: {curr_frame}", (5, height_ration),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                
            crop_images = crop_objects(clean_frame, boxes, crop_path, valid_list, curr_frame, save=True)
            if len(crop_images) > 1:
                #manage case when more than one car detected
                pass
            else:
                car_list.append(Car(img=crop_images[0], time=datetime.datetime.now()))

            cropped_frames += 1
        
        if show:
            cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)    
            cv2.moveWindow("result", 40, 30)
            cv2.imshow('result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):   break
        
        curr_frame += 1
        if write:   writer.write(frame)
        fps = int(1/(time.time()-start_time))
        print(f'FPS: {fps}')
            
    cv2.destroyAllWindows()       
    print('\nDone.')
    return car_list


def main():
    args = parse_args()
    CONFIG = read_json("cfg.json")
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    video_name = args.video.split("/")[-1]
    video_name = video_name.split(".")[0]
    
    if args.output:
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (frame_width, frame_height))
    else:   
        writer = None


    cls_dict = get_cls_dict(CONFIG["CLASSES"])
    d_zone = CONFIG["DETECTION_ZONE"]
    vis = BBoxVisualization(cls_dict, d_zone)
    trt_yolo = TrtYOLO(args.model, 2, args.letter_box)
    args_list = [args.show_vid, args.output, args.detection_zone, args.detect_car]

    cars = loop_and_detect(cap, trt_yolo, conf_th=args.threshold, vis=vis, writer=writer, args=args_list, video_name=video_name)
    key = CONFIG["KEY"]

    if args.output: writer.release()

    cap.release()

    # Get plate number
    for car in cars:
        ocr_res = ocr_api(car.img, key, regions=["ca"])
        pprint(ocr_res)
        # ocr api limits calls every second so add a little delay
        time.sleep(0.1)
        results = ocr_res['results']
        if results:
            print("Plate detected: ", results[0]['plate'])
            car.set_ocr_info(ocr_json=ocr_res)
            break


if __name__ == '__main__':
    main()
