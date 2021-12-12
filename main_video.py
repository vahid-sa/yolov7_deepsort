import os
import argparse
from threading import Thread, enumerate
from queue import Queue
from os import path as osp
from yolov4.darknet_video import VideoDarknet


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not osp.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(osp.abspath(args.config_file))))
    if not osp.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(osp.abspath(args.weights))))
    if not osp.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(osp.abspath(args.data_file))))
    if str2int(args.input) == str and not osp.exists(args.input):
        raise(ValueError("Invalid video path {}".format(osp.abspath(args.input))))
    if not osp.isfile(args.libso):
        raise FileNotFoundError("Invalid libso path {}".format(osp.expanduser(osp.expandvars(osp.abspath(args.libso)))))


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=0.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--libso", type=str, default="./libdarknet.so",
                        help="Path to 'libdarknet.so' file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    input_path = str2int(args.input)
    video_darknet = VideoDarknet(
        input_video=input_path,
        weights_path=args.weights,
        config_path=args.config_file,
        data_file_path=args.data_file,
        libso_path=osp.expanduser(osp.expandvars(osp.abspath(args.libso)))
    )
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    t1 = Thread(
        target=video_darknet.video_capture,
        args=(frame_queue, darknet_image_queue),
    )
    t2 = Thread(
        target=video_darknet.inference,
        args=(darknet_image_queue, detections_queue, fps_queue, args.thresh, args.ext_output),
    )
    t3 = Thread(
        target=video_darknet.drawing,
        args=(frame_queue, detections_queue, fps_queue, args.out_filename, args.dont_show),
    )
    t1.start()
    t2.start()
    t3.start()
