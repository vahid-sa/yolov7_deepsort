"""
This is a demo of using the tracker.
"""


import cv2
import yaml
import time
from os import path as osp

from tracker.detect import Detector
from tracker.track import Tracker
from tracker.draw import Drawer


class Args:
    def __init__(self):
        f = open("configs.yaml")
        opt = yaml.safe_load(f)
        f.close()
        self.input_path = self.complete_path(opt["input"])
        self.output_path = self.complete_path(opt["output"])
        self.weights_path = self.complete_path(opt["weights"])
        self.protbuf_path = self.complete_path(opt["protbuf"])
        self.names = self.complete_path(opt["names"])
        self.device = opt["device"]

    @staticmethod
    def complete_path(p):
        return osp.abspath(osp.expanduser(osp.expandvars(p)))


def main(args: Args):
    cap = cv2.VideoCapture(args.input_path)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    with open(args.names) as f:
        string = f.read()
    class_names = string.splitlines()
    detector = Detector(
        device=args.device,
        img_size=640,
        weights=args.weights_path,
        class_names=class_names,
    )
    tracker = Tracker(protbuf_path=args.protbuf_path)
    drawer = Drawer()
    writer = None
    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(
            args.output_path,
            fourcc,
            int(fps),
            (int(image_width), int(image_height)),
        )
    cv2.namedWindow('Tracker')
    idx = 0
    average_latency = 0.0
    while cap.isOpened():
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        predictions = detector(img0=frame)
        detections, tracks = tracker(frame=frame, predictions=predictions)
        image = drawer(image=frame, detections=detections, tracks=tracks)
        if writer is not None:
            writer.write(image)
        cv2.imshow("Tracker", image)
        print(f"\rFrame {idx}", end="")
        if cv2.waitKey(1) == ord('q'):
            break
        t2 = time.time()
        average_latency = (idx * average_latency + (t2 - t1)) / (idx + 1)
        idx += 1
    if writer is not None:
        writer.release()
    cap.release()
    print(f"\nAverage Latency: {int(average_latency * 1000)} ms\nExited.")


if __name__ == "__main__":
    main(args=Args())
