import math
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F

# from line_profiler_pycharm import profile
from Utils.utils import overtime_reminder_deco, Tools

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = Tools.get_logger("RealESR", '')


class RealESRGANer:
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, half=False):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_block = 23
        net_scale = scale
        is_change_RRDB = False
        is_RFDN = False
        if 'RealESRGAN_x4plus_anime_6B.pth' in model_path:
            num_block = 6
        elif 'RealESRGAN_x2plus.pth' in model_path:
            """Double Check"""
            net_scale = 2
        elif 'RealESRGAN_x2plus_anime110k_6B.pth' in model_path:
            is_change_RRDB = True
            num_block = 6
            net_scale = 2
        elif 'RFDN' in model_path:
            is_RFDN = True
        # debug
        # num_block = 23
        if is_RFDN:
            from basicsr.archs.rfdn_arch import RFDN
            model = RFDN()
        else:
            if is_change_RRDB:
                from basicsr.archs.svfi_rrdbnet_arch import MyRRDBNet as RRDBNet
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet as RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_grow_ch=32,
                            num_block=num_block, scale=net_scale)

        loadnet = torch.load(model_path, map_location='cpu')
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.half().to(self.device)  # compulsory switch to half mode

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        # if self.half:
        self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        self.output = self.model(self.img)

    def tile_process(self):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                # print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class SvfiRealESR:
    def __init__(self, model="", gpu_id=0, precent=90, scale=4, tile=100, resize=(0, 0), half=False):
        # TODO optimize auto detect tilesize
        # const_model_memory_usage = 0.6
        # const_pixel_memory_usage = 0.9 / 65536
        # total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3) * 0.8 * (
        #             precent / 100)
        # available_memory = (total_memory - const_model_memory_usage)
        # tile_size = int(math.sqrt(available_memory / const_pixel_memory_usage)) / scale * 2
        # padding = (scale ** 2) * tile_size
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.resize_param = resize
        self.scale_exp = 4
        self.scale = scale
        # self.scale = 4

        # Clarify Model Scale
        if "x4plus" in model:
            self.scale_exp = 4
        elif "x2plus" in model:
            self.scale_exp = 2

        self.available_scales = [2, 4]
        self.alpha = None
        model_path = os.path.join(app_dir, "ncnn", "sr", "realESR", "models", model)
        self.upscaler = RealESRGANer(scale=self.scale_exp, model_path=model_path, tile=tile, half=half)
        pass

    def resize_esr_img(self, img):
        """
        # RealESR 2x, 4x目标分辨率转化

        :param img:
        :return: resized img
        """
        w, h = self.resize_param
        resize_width = int(w / self.scale_exp)
        resize_height = int(h / self.scale_exp)
        if int(resize_width) % 2:
            resize_width += 1
        if int(resize_height) % 2:
            resize_height += 1
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
        return img

    # @profile
    @overtime_reminder_deco(300, logger, "RealESR",
                            "Low Super-Resolution speed detected, Please Consider tweak tilesize to enhance speed")
    def svfi_process(self, img):
        if all(self.resize_param):
            img = self.resize_esr_img(img)
        if self.scale > 1:
            cur_scale = 1
            while cur_scale < self.scale:
                img = self.process(img)
                cur_scale *= self.scale_exp
        if all(self.resize_param):
            img = cv2.resize(img, self.resize_param, interpolation=cv2.INTER_LANCZOS4)
        return img

    def process(self, img):
        output, img_mode = self.upscaler.enhance(img)
        return output


if __name__ == '__main__':
    test = SvfiRealESR(model="RealESRGAN_x4plus_anime_6B.pth", )
    # test.svfi_process(cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\Utils\RealESRGAN\input\used\input.png"))
    o = test.svfi_process(
        cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\0.png", cv2.IMREAD_UNCHANGED))
    cv2.imwrite("out2.png", o)
