from networks.deeplab.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm):
    return resnet.ResNet101(output_stride, BatchNorm)
