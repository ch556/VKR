import torch
import numpy as np
from OCR.tokenizer import Tokenizer
from model import CRNN
from transforms import InferenceTransform

config_json = {
    "alphabet": " !\"'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPRSTVWY[\\]_abcdefghiklmnoprstuvwxyz|}ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё’№",
    "image": {
        "width": 512,
        "height": 64
    }
}


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


class OcrPredictor:
    def __init__(self, model_path, config, device='cpu'):
        self.tokenizer = Tokenizer(config['alphabet'])
        self.device = torch.device(device)

        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars(), pretrained=True)
        for name, param in self.model.named_parameters():  #FIXME
            if name == 'classifier.3.weight':
                param.data = torch.nn.Parameter(torch.randn(148, 256))
            elif name == 'classifier.3.bias':
                param.data = torch.nn.Parameter(torch.randn(148, ))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config['image']['height'],
            width=config['image']['width'],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


def init_ocr_predictor():
    predictor = OcrPredictor(
        model_path='ocr_model.ckpt',
        config=config_json,
        device='cpu'
    )
    return predictor

