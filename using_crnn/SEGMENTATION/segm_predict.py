from segm_transforms import get_contours_from_mask, rescale_contours, reduce_contours_dims, contour2bbox
import torch
def predict(images, model, device, targets=None):
    """Make model prediction.
    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        device (torch.device): Torch device.
        targets (torch.Tensor): Batch with tensor masks. By default is None.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)

    if targets is not None:
        targets = targets.to(device)
        return output, targets
    return output


def get_preds(images, preds, cls2params, config, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):  # iterate through images
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}
        for cls_idx, cls_name in enumerate(cls2params):  # iter through classes
            pred_cls = pred[cls_idx]
            # thresholding works faster on cuda than on cpu
            pred_cls = \
                pred_cls > cls2params[cls_name]['postprocess']['threshold']
            if cuda_torch_input:
                pred_cls = pred_cls.cpu().numpy()

            contours = get_contours_from_mask(
                pred_cls, cls2params[cls_name]['postprocess']['min_area'])
            contours = rescale_contours(
                contours=contours,
                pred_height=config.get_image('height'),  #FIXME
                pred_width=config.get_image('width'),
                image_height=img_h,
                image_width=img_w
            )
            bboxes = [contour2bbox(contour) for contour in contours]
            contours = reduce_contours_dims(contours)

            for contour, bbox in zip(contours, bboxes):
                pred_img['predictions'].append(
                    {
                        'polygon': contour,
                        'bbox': bbox,
                        'class_name': cls_name
                    }
                )
        pred_data.append(pred_img)
    return pred_data