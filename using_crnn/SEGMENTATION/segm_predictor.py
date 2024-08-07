from segm_model import SegmTorchModel

class SegmPredictor:
    def __init__(self, model_path, config_path, device):
        self.model = SegmTorchModel(model_path, config_path, device)

    def __call__(self, images):
        preds = self.model.predict(images)
        pred_data = self.model.get_preds(images, preds)
        return pred_data

import os
def init_segm_predictor():
    cfg_path = os.path.join('SEGMENTATION', 'segm_config.json')
    model_path = os.path.join('SEGMENTATION', 'segm_model.ckpt')
    assert os.path.exists(model_path), 'segm_model.ckpt does not exist!'
    assert os.path.exists(cfg_path), f'{cfg_path} does not exist!'
    predictor = SegmPredictor(
        model_path=model_path,
        config_path=cfg_path,
        device='cpu', )
    return predictor





