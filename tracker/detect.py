import torch
import cv2
import numpy as np
from typing import Tuple

from .yolov7.models.experimental import attempt_load
from .yolov7.utils.general import non_max_suppression, scale_coords
from .yolov7.utils.torch_utils import TracedModel, select_device
from .yolov7.utils.datasets import letterbox


class Detector:
    """Object Detection using YOLOv7"""

    def __init__(
        self,
        device: str,
        weights: str,
        img_size: int,
        class_names: list,
        trace: bool = True,
        half: bool = True,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        agnostic_nms: bool = False,
    ):
        """Class Constructor

        Args:
            device (str): Specify cpu or gpu; cpu; cuda:0; cuda:3; 0,1,2; ...
            weights (str): Path to the weights file
            img_size (int): frame width and height will be resized to this value. Refer to readme
            class_names (list): name of classes. the order must be compatible with the weights.
            trace (bool, optional): YOLOv7 trace model. Defaults to True.
            half (bool, optional): For gpu if True will be more efficient by time and memory. Defaults to True.
            confidence_threshold (float, optional): Objects with lower confidence than the threshold will be ignored. Defaults to 0.25.
            iou_threshold (float, optional): IoU of the predicted over ground-truth will be ignored for lower than threshold. Defaults to 0.45.
            agnostic_nms (bool, optional): Refer to YOLOv7 repo. Defaults to False.
        """
        self._device = self._select_device(device)
        self._half = half if (self._device.type == 'cuda') else False
        self._model = attempt_load(weights=weights, map_location=self._device)
        self._conf_thres = confidence_threshold
        self._iou_thres = iou_threshold
        self._agnostic_nms = agnostic_nms
        self._img_size = img_size
        self._class_names = class_names
        if trace:
            self._model = TracedModel(self._model, device, self._img_size)
        if self._half:
            self._model.half()
        self._stride = int(self._model.stride.max())

    def __call__(self, img0: np.ndarray) -> Tuple[np.ndarray, list, list]:
        """Detects objects in the image

        Args:
            img0 (np.ndarray): Input img from IO (BGR)

        Returns:
            Tuple[np.ndarray, list, list]: bboxes (xyxy), confidences for each detection, class of each detection
        """
        img = self._preprocess_img(img0)
        img: torch.Tensor = torch.from_numpy(img).to(self._device)
        with torch.no_grad():
            pred = self._model(img, augment=False)[0]
        pred = pred.cpu().float()
        pred = non_max_suppression(
            pred,
            self._conf_thres,
            self._iou_thres,
            classes=None,
            agnostic=self._agnostic_nms,
        )
        det = pred[0]
        if len(det) > 0:
            det[:, :4] = scale_coords(
                img1_shape=img.shape[2:],
                coords=det[:, :4],
                img0_shape=img0.shape[:2],
            ).round()
            bboxes = det[:, :4].int().numpy()
            confidences = det[:, 4].tolist()
            classes = [self._class_names[index] for index in det[:, 5].int()]
        return bboxes, confidences, classes

    def _preprocess_img(self, img0: np.ndarray) -> np.ndarray:
        """Prepares the img for loading on the model.

        Args:
            img0 (np.ndarray): Frame taken from IO.

        Returns:
            np.ndarray: Frame ready to load.
        """
        img = img0
        img = letterbox(img, self._img_size, stride=self._stride)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = img.astype('float16') if self._half else img.astype('float32')
        img /= 255.0
        return img

    def _select_device(self, device: str) -> torch.device:
        """which device to load Tensors on and proceesing in.

        Args:
            device (str): 'cpu' or 'cuda:{n}' or 'i,j,k': index of devices

        Returns:
            torch.device: selected device on torch class.
        """
        if (device == "cpu") or (not torch.cuda.is_available()):
            selected_device = select_device("cpu")
        elif device == "cuda":
            selected_device = select_device("0")
        elif device.startswith("cuda:"):
            selected_device = select_device(device[5:])
        else:
            selected_device = select_device(device)
        return selected_device
