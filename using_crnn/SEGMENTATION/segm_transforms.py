import cv2
import numpy as np
import torch
import torchvision


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in
    pytorch (NxCxHxW)."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class Scale:
    def __init__(self, height, width):
        self.size = (width, height)

    def __call__(self, img, mask=None):
        resize_img = cv2.resize(img, self.size, cv2.INTER_LINEAR)
        if mask is not None:
            resize_mask = cv2.resize(mask, self.size, cv2.INTER_LINEAR)
            return resize_img, resize_mask
        return resize_img


class InferenceTransform:
    def __init__(self, height, width, return_numpy=False):
        self.transforms = torchvision.transforms.Compose([
            Scale(height, width),
            MoveChannels(to_channels_first=True),
            Normalize(),
        ])
        self.return_numpy = return_numpy
        self.to_tensor = ToTensor()

    def __call__(self, images):
        transformed_images = [self.transforms(image) for image in images]
        transformed_array = np.stack(transformed_images, 0)
        if not self.return_numpy:
            transformed_array = self.to_tensor(transformed_array)
        return transformed_array


def contour2bbox(contour):
    """Get bbox from contour."""
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def rescale_contours(
    contours, pred_height, pred_width, image_height, image_width
):
    """Rescale contours from prediction mask shape to input image size."""
    y_ratio = image_height / pred_height
    x_ratio = image_width / pred_width
    scale = (x_ratio, y_ratio)
    for contour in contours:
        for i in range(2):
            contour[:, :, i] = contour[:, :, i] * scale[i]
    return contours


def reduce_contours_dims(contours):
    reduced_contours = []
    for contour in contours:
        contour = [[int(i[0][0]), int(i[0][1])] for i in contour]
        reduced_contours.append(contour)
    return reduced_contours