import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from models.layers import build_cnn
from models.layers import GlobalAvgPool
from models.bilinear import crop_bbox_batch
# from models.initialization import weights_init
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
device = torch.device('cuda:0')


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def _downsample(x):
    return F.avg_pool2d(x, kernel_size=2)


class OptimizedBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.learnable_sc = (dim_in != dim_out) or downsample
        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        h = x
        if self.downsample:
            h = _downsample(x)
        return self.sc(h)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.learnable_sc = (dim_in != dim_out) or downsample

        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ImageDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(ImageDiscriminator, self).__init__()
        self.ch = conv_dim
        self.relu = nn.ReLU(inplace=True)
        self.main = nn.Sequential(
            # (3, 64, 64) -> (64, 32, 32)
            OptimizedBlock(3, self.ch, downsample=True),
            # (64, 32, 32) -> (128, 16, 16)
            ResidualBlock(self.ch, self.ch * 2, downsample=True),
            # (128, 16, 16) -> (256, 8, 8)
            ResidualBlock(self.ch * 2, self.ch * 4, downsample=True),
            # (256, 8, 8) -> (512, 4, 4)
            ResidualBlock(self.ch * 4, self.ch * 8, downsample=True),
            # (512, 4, 4) -> (1024, 2, 2)
            ResidualBlock(self.ch * 8, self.ch * 16, downsample=True),
        )

        self.classifier = nn.Linear(self.ch * 16, 1, bias=False)

        # self.apply(weights_init)

    def forward(self, x):
        h = self.main(x)
        h = self.relu(h)
        # (1024, 2, 2) -> (1024,)
        h = torch.sum(h, dim=(2, 3))

        output = self.classifier(h)

        return output.view(-1)


class ObjectDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, n_class=0, downsample_first=False):
        super(ObjectDiscriminator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.main = nn.Sequential(
            # (3, 32, 32) -> (64, 32, 32)
            OptimizedBlock(3, conv_dim, downsample=downsample_first),
            # (64, 32, 32) -> (128, 16, 16)
            ResidualBlock(conv_dim, conv_dim * 2, downsample=True),
            # (128, 16, 16) -> (256, 8, 8)
            ResidualBlock(conv_dim * 2, conv_dim * 4, downsample=True),
            # (256, 8, 8) -> (512, 4, 4)
            ResidualBlock(conv_dim * 4, conv_dim * 8, downsample=True),
            # (512, 4, 4) -> (1024, 2, 2)
            ResidualBlock(conv_dim * 8, conv_dim * 16, downsample=True),
        )

        self.classifier_src = nn.Linear(conv_dim * 16, 1)
        self.classifier_cls = nn.Linear(conv_dim * 16, n_class)

        # if n_class > 0:
        #     self.l_y = nn.Embedding(num_embeddings=n_class, embedding_dim=conv_dim * 16)

        # self.apply(weights_init)

    def forward(self, x, y=None):
        h = x
        h = self.main(h)
        h = self.relu(h)
        # (1024, 2, 2) -> (1024,)
        h = torch.sum(h, dim=(2, 3))

        output_src = self.classifier_src(h)
        output_cls = self.classifier_cls(h)

        return output_src.view(-1), output_cls


if __name__ == '__main__':
    dataset_file = '../data/dataset_vocab_109_objnum_1_10_imgnum_100382_train.json'

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    id2vocab = dataset['id2vocab']
    vocab2id = dataset['vocab2id']
    images = dataset['images']

    image0 = images[0]
    image1 = images[1]

    image_id = [image0['image_id'], image1['image_id']]
    object_ids = np.asarray([image0['object_ids'], image1['object_ids']])
    object_ids = torch.LongTensor(object_ids).to(device)

    object_bboxes = np.asarray([image0['object_bboxes'], image1['object_bboxes']])
    object_bboxes = torch.Tensor(object_bboxes)

    object_ids_flat = []
    object_bboxes_flat = []
    object_to_image_flat = []
    for i in range(2):
        idx = (object_ids[i] > 0).nonzero().view(-1)

        for j in idx:
            object_ids_flat.append(object_ids[i, j])
            object_bboxes_flat.append(object_bboxes[i, j])
            object_to_image_flat.append(i)

    object_ids_flat = torch.LongTensor(object_ids_flat).to(device)
    object_bboxes_flat = torch.stack(object_bboxes_flat, 0).to(device)
    object_to_image_flat = torch.LongTensor(object_to_image_flat).to(device)

    input = torch.rand(2, 3, 64, 64).to(device)

    D_img = ImageDiscriminator(embed_dim=64).to(device)

    D_obj = ObjectDiscriminator(num_vocab=10, embed_dim=64, object_size=32).to(device)

    D_img = add_sn(D_img)
    D_obj = add_sn(D_obj)

    output = D_img(input)
    print(output.shape)

    #  imgs, objs, boxes, obj_to_img
    #  Inputs:
    #     - feats: FloatTensor of shape (N, C, H, W)
    #     - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    #     - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
    #       each element is in the range [0, N) and bbox_to_feats[b] = i means that
    #       bbox[b] will be cropped from feats[i].
    #     - HH, WW: Size of the output crops
    feats = torch.zeros(2, 3, 64, 64).to(device)
    # bbox = torch.zeros(6, 4).to(device)
    # objs = torch.LongTensor([0, 0, 0, 1, 2, 3]).to(device)
    # bbox_to_feats = torch.LongTensor([0, 0, 0, 1, 1, 1]).to(device)
    # HH, WW = 64, 64

    output = D_obj(feats, object_ids_flat, object_bboxes_flat, object_to_image_flat)

    print(output.shape)
