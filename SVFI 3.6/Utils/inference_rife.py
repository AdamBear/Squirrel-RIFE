import os
import traceback
import warnings

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from Utils.utils import ArgumentManager, appDir, VideoFrameInterpolation

warnings.filterwarnings("ignore")


class RifeInterpolation(VideoFrameInterpolation):
    def __init__(self, __args: ArgumentManager):
        super().__init__(__args)
        self.initiated = False
        self.ARGS = __args

        self.use_auto_scale = self.ARGS.use_rife_auto_scale
        self.auto_scale_predict_size = self.ARGS.rife_auto_scale_predict_size
        self.device = None
        self.device_count = torch.cuda.device_count()
        self.model = None
        self.model_path = ""
        self.model_version = 0
        self.tta_mode = self.ARGS.rife_tta_mode
        self.tta_iter = self.ARGS.rife_tta_iter

    def initiate_algorithm(self, __args=None):
        if self.initiated:
            return

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.ARGS.use_rife_fp16 = False
            print("INFO - use cpu to interpolate")
        else:
            self.device = torch.device(f"cuda")
            # torch.cuda.set_device(self.ARGS.use_specific_gpu)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            if self.ARGS.use_rife_fp16:
                try:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
                    print("INFO - FP16 mode switch success")
                except Exception as e:
                    print("INFO - FP16 mode switch failed")
                    traceback.print_exc()
                    self.ARGS.use_rife_fp16 = False

        if self.ARGS.rife_model == "" or not os.path.exists(self.ARGS.rife_model):
            self.model_path = os.path.join(appDir, 'train_log', 'official_2.3')
        else:
            self.model_path = self.ARGS.rife_model

        try:
            from model.RIFE_HDv2 import Model
            model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                          forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
            model.load_model(self.model_path, -1 if not self.ARGS.use_rife_multi_cards else 0)
            self.model_version = 2
            print("INFO - Loaded v2.x HD model.")
        except:
            from model.RIFE_HDv3 import Model
            model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                          forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
            model.load_model(self.model_path, -1)
            self.model_version = 3
            print("INFO - Loaded v3.x HD model.")

        self.model = model
        self.model.eval()
        self.model.device()
        self.initiated = True

    def __inference(self, i1, i2, scale):
        if self.ARGS.is_rife_reverse:
            mid = self.model.inference(i1, i2, scale, iter_time=self.tta_iter)
        else:
            mid = self.model.inference(i2, i1, scale, iter_time=self.tta_iter)
        return mid

    def __make_n_inference(self, img1, img2, scale, n):
        padding, h, w = self.generate_padding(img1, scale)
        i1 = self.generate_torch_img(img1, padding)
        i2 = self.generate_torch_img(img2, padding)
        mid = self.__inference(i1, i2, scale)
        del i1, i2
        mid = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w].copy()
        if n == 1:
            return [mid]
        first_half = self.__make_n_inference(img1, mid, scale, n=n // 2)
        second_half = self.__make_n_inference(mid, img2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def generate_padding(self, img, scale: float):
        """

        :param scale:
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return padding, h, w

    def generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """

        """
        Multi Cards Optimization:
        OLS: send several imgs pair according to device_count (2 to be specific)
        HERE: Concat [i1, i2] [i3, i4] and send to rife
        """

        try:
            img_torch = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0)
            if self.ARGS.use_rife_fp16:
                img_torch = img_torch.half() / 255.
            else:
                img_torch = img_torch.float() / 255.
            if self.ARGS.use_rife_multi_cards and self.device_count > 1:
                if self.device_count % 2 == 0:
                    batch = 2
                else:
                    batch = 3
                img_torch = torch.cat([img_torch for i in range(batch)], dim=0)
            return self.pad_image(img_torch, padding)
        except Exception as e:
            print(img)
            traceback.print_exc()
            raise e

    def pad_image(self, img, padding):
        # if self.ARGS.use_rife_fp16:
        #     return F.pad(img, padding).half()
        # else:
        return F.pad(img, padding)

    def generate_n_interp(self, img1, img2, n, scale, debug=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen

        interp_gen = self.__make_n_inference(img1, img2, scale, n=n)
        return interp_gen

    def _get_auto_scale_to_tensor(self, *imgs):
        if self.ARGS.use_rife_fp16:  # 是否为半精度
            return [
                torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).half() / 255.
                for img in imgs]
        return [torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                for img in imgs]

    def get_auto_scale(self, i0, i1):
        scale_range = [0.25, 0.5, 1.0]
        max_distance = 0
        select_scale = 0.5
        for scale in scale_range:
            pwh = int(32/scale)
            t0 = cv2.resize(i0, (pwh, pwh))
            t1 = cv2.resize(i1, (pwh, pwh))
            I0, I1 = self._get_auto_scale_to_tensor(t0, t1)
            lf, _ = self.model.calculate_flow(I0, I1, scale)
            rf, _ = self.model.calculate_flow(I1, I0, scale)
            distance = abs(((lf - rf) / 2).mean())
            if distance > max_distance:
                select_scale = scale
                max_distance = distance
        return select_scale

    def run(self):
        pass


if __name__ == "__main__":
    pass
