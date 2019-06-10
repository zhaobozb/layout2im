import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_bbox_batch(images, bbox_sets):
    device = images.device
    results = []
    images = images.cpu().numpy()
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)), dtype=np.float32)
    for image, bbox_set in zip(images, bbox_sets):
        for bbox in bbox_set:
            if all(bbox == 0):
                continue
            else:
                image = draw_bbox(image, bbox)

        results.append(image)

    images = np.stack(results, axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).float().to(device)
    return images


def draw_bbox(image, bbox):
    im = Image.fromarray(np.uint8(image * 255))
    draw = ImageDraw.Draw(im)

    h, w, _ = image.shape
    c1 = (round(float(bbox[0] * w)), round(float(bbox[1] * h)))
    c2 = (round(float(bbox[2] * w)), round(float(bbox[3] * h)))

    draw.rectangle([c1, c2], outline=(0, 255, 0))

    output = np.array(im)/255

    return output

def str2bool(v):
    return v.lower() == 'true'