# coding: utf-8

import cv2
import numpy as np
from PIL import Image

from Utils.utils import overtime_reminder_deco, Tools, appDir
from ncnn.sr.realSR.realsr_ncnn_vulkan import RealSR
from ncnn.sr.waifu2x.waifu2x_ncnn_vulkan import Waifu2x

logger = Tools.get_logger('SWIG-SR', appDir)


class SvfiWaifu(Waifu2x):
    def __init__(self, model="", scale=1, num_threads=4, resize=(0, 0), **kwargs):
        self.available_scales = [2, 4, 8, 16]
        super().__init__(gpuid=0,
                         model=model,
                         tta_mode=False,
                         num_threads=num_threads,
                         scale=scale,
                         noise=0,
                         tilesize=0, )
        self.resize_param = resize

    @overtime_reminder_deco(300, logger, "RealSR",
                            "Low Super-Resolution speed detected, Please Consider lower your output resolution to enhance speed")
    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if all(self.resize_param):
            image = cv2.resize(img, self.resize_param, interpolation=cv2.INTER_LANCZOS4)
        return image


class SvfiRealSR(RealSR):
    def __init__(self, model="", scale=1, num_threads=4, resize=(0, 0), **kwargs):
        self.available_scales = [4, 16]
        super().__init__(gpuid=0,
                         model=model,
                         tta_mode=False,
                         scale=scale,
                         tilesize=0, )
        self.resize_param = resize

    @overtime_reminder_deco(300, logger, "RealSR",
                            "Low Super-Resolution speed detected, Please Consider lower your output resolution to enhance speed")
    def svfi_process(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self.process(image)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if all(self.resize_param):
            image = cv2.resize(img, self.resize_param, interpolation=cv2.INTER_LANCZOS4)
        return image
