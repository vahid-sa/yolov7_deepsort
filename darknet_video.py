from ctypes import *
import random
import cv2
import time
from darknet import DarkNet

class VideoDarknet:
    def __init__(self, input_video, weights_path, config_path, data_file_path, libso_path):
        self.darknet = DarkNet(libso_path)
        self.network, self.class_names, self.class_colors = self.darknet.load_network(config_path, data_file_path, weights_path, batch_size=1)
        self.darknet_width = self.darknet.network_width(self.network)
        self.darknet_height = self.darknet.network_height(self.network)
        self.cap = cv2.VideoCapture(input_video)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def set_saved_video(input_video, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        print("\ninput: {0}\noutput: {1}\nfourcc: {2}\nfps: {3}\nsize: {4}\n".format(input_video, output_video, fourcc, fps, size))
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video

    def convert2relative(self, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h  = bbox
        _height     = self.darknet_height
        _width      = self.darknet_width
        return x/_width, y/_height, w/_width, h/_height

    def convert2original(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_x       = int(x * image_w)
        orig_y       = int(y * image_h)
        orig_width   = int(w * image_w)
        orig_height  = int(h * image_h)

        bbox_converted = (orig_x, orig_y, orig_width, orig_height)

        return bbox_converted

    def convert4cropping(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

        return bbox_cropping

    def video_capture(self, frame_queue, darknet_image_queue):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame)
            img_for_detect = self.darknet.make_image(self.darknet_width, self.darknet_height, 3)
            self.darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)
        self.cap.release()

    def inference(self, darknet_image_queue, detections_queue, fps_queue, threshold: float=0.25, show_coordinates: bool=False):
        while self.cap.isOpened():
            darknet_image = darknet_image_queue.get()
            prev_time = time.time()
            detections = self.darknet.detect_image(self.network, self.class_names, darknet_image, thresh=threshold)
            detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            fps_queue.put(fps)
            print("FPS: {}".format(fps))
            self.darknet.print_detections(detections, show_coordinates)
            self.darknet.free_image(darknet_image)
        self.cap.release()

    def drawing(self, frame_queue, detections_queue, fps_queue, output_video_path=None, dont_show: bool = False):
        random.seed(3)  # deterministic bbox colors
        video = VideoDarknet.set_saved_video(self.cap, output_video_path, (self.video_width, self.video_height))
        while self.cap.isOpened():
            frame = frame_queue.get()
            detections = detections_queue.get()
            fps = fps_queue.get()
            detections_adjusted = []
            if frame is not None:
                for label, confidence, bbox in detections:
                    bbox_adjusted = self.convert2original(frame, bbox)
                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
                image = self.darknet.draw_boxes(detections_adjusted, frame, self.class_colors)
                if not dont_show:
                    cv2.imshow('Inference', image)
                if output_video_path is not None:
                    video.write(image)
                if cv2.waitKey(fps) == 27:
                    break
        video.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.cap.release()
