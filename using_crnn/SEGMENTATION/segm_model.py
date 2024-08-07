import torch.nn as nn
from torch.nn import Conv2d
from torchvision.models.resnet import resnet50
from segm_config import Config
import torch
from segm_transforms import InferenceTransform
from segm_predict import predict, get_preds

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



ENCODERS = {
    'resnet50': resnet50,
}

class LinkResNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, dropout2d_p=0.5,
                 pretrained=True, encoder='resnet50'):
        assert input_channels > 0

        super().__init__()

        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]

        resnet = ENCODERS[encoder](pretrained=pretrained)
        if input_channels != 3:
            resnet.conv1 = Conv2d(input_channels, 64, kernel_size=(7, 7),
                                  stride=(2, 2), padding=(3, 3), bias=False)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dropout2d1 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d2 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d3 = nn.Dropout2d(p=dropout2d_p)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, output_channels, 2, padding=1)
        self.sigmoid = nn.Sigmoid()

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + self.dropout2d1(e3)
        # d4 = e3
        d3 = self.decoder3(d4) + self.dropout2d2(e2)
        d2 = self.decoder2(d3) + self.dropout2d3(e1)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return self.sigmoid(f5)

class SegmTorchModel():
    def __init__(self, model_path, config_path, device='cuda'):
        self.config = Config(config_path)
        self.device = torch.device(device)
        self.cls2params = self.config.get_classes()
        # load model
        self.model = LinkResNet(
            output_channels=len(self.cls2params),
            pretrained=False
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        preds = predict(transformed_images, self.model, self.device)
        return preds

    def get_preds(self, images, preds):
        pred_data = get_preds(images, preds, self.cls2params, self.config)
        return pred_data