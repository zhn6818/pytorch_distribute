#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )

# class RandomCrop(object):
#     def __init__(self, size, *args, **kwargs):
#         self.size = size

#     def __call__(self, im_lb):
#         im = im_lb['im']
#         lb = im_lb['lb']
#         assert im.size == lb.size
#         im_np = np.array(im)
#         lb_np = np.array(lb)
#         tw, th = self.size
#         h, w = im_np.shape[0:2]
#         if w == tw and h == th:
#             return dict(im=im, lb=lb)
#         if w < tw and h < th :
#             return dict(im=im.resize((tw,th),Image.BILINEAR),lb=lb.resize((tw,th),Image.NEAREST))

#         if random.random() > 3.0 / 8.0 and np.max(lb_np) > 0:
#             tl = np.min(np.where(lb_np > 0), axis = 1) - (th,tw)
#             tl[tl < 0] = 0
#             br = np.max(np.where(lb_np > 0), axis = 1) - (th,tw)
#             br[br < 0] = 0
#             br[0] = min(br[0], h - th)
#             br[1] = min(br[1], w - tw)
#             i = random.randint(tl[0], br[0])
#             j = random.randint(tl[1], br[1])
#         else:
#             if h - th < 0 or w - tw < 0:
#                 return dict(im=im.resize((tw,th),Image.BILINEAR),lb=lb.resize((tw,th),Image.NEAREST))
#             i = random.randint(0, h - th)
#             j = random.randint(0, w - tw)
#         im_np = im_np[i:i + th, j:j + tw, :]
#         lb_np = lb_np[i:i + th, j:j + tw]
#         im = Image.fromarray(im_np)
#         lb = Image.fromarray(lb_np)
#         return dict(
#                 im = im,
#                 lb = lb
#                     )

class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        # if min(H, W) * scale <= 640:
        #     scale = (640 + 10) * 1.0 / min(H, W)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb




if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
