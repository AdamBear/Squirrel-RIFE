# coding: utf-8
import datetime
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import signal
import string
import subprocess
import threading
import time
import traceback
from collections import deque
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np
import psutil
from sklearn import linear_model

from skvideo.utils import check_output
from steamworks import STEAMWORKS
from steamworks.exceptions import *

abspath = os.path.abspath(__file__)
appDir = os.path.dirname(os.path.dirname(abspath))


class AiModulePaths:
    """Relevant to root dir(app dir)"""
    sr_algos = ["ncnn/sr", ]


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov']


class EncodePresetAssemply:
    encoder = {
        "CPU": {
            "H264,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
             "H265,8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
             "H265,10bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"], },
        "NVENCC":
            {"H264,8bit": ["default", "performance", "quality"],
             "H265,8bit": ["default", "performance", "quality"],
             "H265,10bit": ["default", "performance", "quality"], },
        "QSVENCC":
            {"H264,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,10bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },
        "SVT":
            {"VP9,8bit": ["slowest", "slow", "fast", "faster"],
             "H265,8bit": ["slowest", "slow", "fast", "faster"],
             "H265,10bit": ["slowest", "slow", "fast", "faster"], },

    }


class SettingsPresets:
    genre_2 = {(0, 0, 0): {"render_crf": 16}}


class DefaultConfigParser(ConfigParser):
    """
    自定义参数提取
    """

    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value


class Tools:
    resize_param = (480, 270)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        self.resize_param = (480, 270)
        self.crop_param = (0, 0, 0, 0)
        pass

    @staticmethod
    def fillQuotation(string):
        if string[0] != '"':
            return f'"{string}"'
        else:
            return string

    @staticmethod
    def get_logger(name, log_path, debug=False):
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
        if ArgumentManager.is_release:
            logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(levelname)s - %(message)s')

        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.log")

        txt_handler = logging.FileHandler(logger_path, encoding='utf-8')
        txt_handler.setFormatter(logger_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    @staticmethod
    def make_dirs(dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    @staticmethod
    def gen_next(gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    @staticmethod
    def dict2Args(d: dict):
        args = []
        for key in d.keys():
            args.append(key)
            if len(d[key]):
                args.append(d[key])
        return args

    @staticmethod
    def clean_parsed_config(args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                print(f"Warning: Find Empty Args at '{a}'")
                args[a] = ""
        return args
        pass

    @staticmethod
    def check_pure_img(img1):
        if np.var(img1) < 10:
            return True
        return False

    @staticmethod
    def check_non_ascii(s: str):
        ascii_set = set(string.printable)
        _s = ''.join(filter(lambda x: x in ascii_set, s))
        if s != _s:
            return True
        else:
            return False

    @staticmethod
    def get_norm_img(img1, resize=True):
        if resize:
            img1 = cv2.resize(img1, Tools.resize_param, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img1

    @staticmethod
    def get_norm_img_diff(img1, img2, resize=True) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return: float
        """
        img1 = Tools.get_norm_img(img1, resize)
        img2 = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        diff = cv2.absdiff(img1, img2).mean()
        return diff

    @staticmethod
    def get_norm_img_flow(img1, img2, resize=True, flow_thres=1) -> (int, np.array):
        """
        Normalize Difference
        :param flow_thres: 光流移动像素长
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return:  (int, np.array)
        """
        prevgray = Tools.get_norm_img(img1, resize)
        gray = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 绘制线
        step = 10
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []
        flow_cnt = 0

        for l in lines:
            if math.sqrt(math.pow(l[0][0] - l[1][0], 2) + math.pow(l[0][1] - l[1][1], 2)) > flow_thres:
                flow_cnt += 1
                line.append(l)

        cv2.polylines(prevgray, line, 0, (0, 255, 255))
        comp_stack = np.hstack((prevgray, gray))
        return flow_cnt, comp_stack

    @staticmethod
    def get_filename(path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def get_mixed_scenes(img0, img1, n):
        """
        return n-1 images
        :param img0:
        :param img1:
        :param n:
        :return:
        """
        step = 1 / n
        beta = 0
        output = list()
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output

    @staticmethod
    def get_fps(path: str):
        """
        Get Fps from path
        :param path:
        :return: fps float
        """
        if not os.path.isfile(path):
            return 0
        try:
            if not os.path.isfile(path):
                input_fps = 0
            else:
                input_stream = cv2.VideoCapture(path)
                input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            return input_fps
        except Exception:
            return 0

    @staticmethod
    def get_existed_chunks(project_dir: str):
        chunk_paths = []
        for chunk_p in os.listdir(project_dir):
            if re.match("chunk-\d+-\d+-\d+\.\w+", chunk_p):
                chunk_paths.append(chunk_p)

        if not len(chunk_paths):
            return chunk_paths, -1, -1

        chunk_paths.sort()
        last_chunk = chunk_paths[-1]
        chunk_cnt, last_frame = re.findall('chunk-(\d+)-\d+-(\d+).*?', last_chunk)[0]
        return chunk_paths, int(chunk_cnt), int(last_frame)

    @staticmethod
    def popen(args: str):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        p = subprocess.Popen(args, startupinfo=startupinfo)
        return p

    @staticmethod
    def md5(d: str):
        m = hashlib.md5(d.encode(encoding='utf-8'))
        return m.hexdigest()

    @staticmethod
    def get_pids():
        """
        get key-value of pids
        :return: dict {pid: pid-name}
        """
        pid_dict = {}
        pids = psutil.pids()
        for pid in pids:
            p = psutil.Process(pid)
            pid_dict[pid] = p.name()
            # print("pid-%d,pname-%s" %(pid,p.name()))
        return pid_dict

    @staticmethod
    def kill_svfi_related():
        pids = Tools.get_pids()
        for pid, pname in pids.items():
            if pname in ['ffmpeg.exe', 'ffprobe.exe', 'one_line_shot_args.exe', 'QSVEncC64.exe', 'NVEncC64.exe',
                         'SvtHevcEncApp.exe']:
                os.kill(pid, signal.SIGABRT)
                print(f"Warning: Kill Process before exit: {pname}")


class ImgSeqIO:
    def __init__(self, folder=None, is_read=True, thread=4, is_tool=False, start_frame=0, logger=None,
                 output_ext=".png", exp=2, resize=(0, 0), is_esr=False, **kwargs):
        # TODO 解耦成Input，Output，Tool三个类
        if logger is None:
            self.logger = Tools.get_logger(name="ImgIO", log_path=folder)
        else:
            self.logger = logger

        if output_ext[0] != ".":
            output_ext = "." + output_ext
        self.output_ext = output_ext

        if folder is None or os.path.isfile(folder):
            self.logger.error(f"Invalid ImgSeq Folder: {folder}")
            return
        self.seq_folder = folder  # + "/tmp"  # weird situation, cannot write to target dir, father dir instead
        if not os.path.exists(self.seq_folder):
            os.makedirs(self.seq_folder, exist_ok=True)
            start_frame = 0
        elif start_frame == -1 and not is_read:
            # write: start writing at the end of sequence
            start_frame = self.get_write_start_frame()
        elif start_frame != -1 and is_read:
            start_frame = int(start_frame / (2 ** exp))

        self.start_frame = start_frame
        self.frame_cnt = 0
        self.img_list = list()

        self.write_queue = Queue(maxsize=1000)
        self.thread_cnt = thread
        self.thread_pool = list()

        self.use_imdecode = False
        self.resize = resize
        self.resize_flag = all(self.resize)
        self.is_esr = is_esr

        self.exp = exp

        if is_tool:
            return
        if is_read:
            img_list = os.listdir(self.seq_folder)
            img_list.sort()
            for p in img_list:
                fn, ext = os.path.splitext(p)
                if ext.lower() in SupportFormat.img_inputs:
                    if self.frame_cnt < start_frame:
                        self.frame_cnt += 1  # update frame_cnt
                        continue  # do not read frame until reach start_frame img
                    self.img_list.append(os.path.join(self.seq_folder, p))
            self.logger.debug(f"Load {len(self.img_list)} frames at {self.frame_cnt}")
        else:
            """Write Img"""
            self.frame_cnt = start_frame
            self.logger.debug(f"Start Writing {self.output_ext} at No. {self.frame_cnt}")
            for t in range(self.thread_cnt):
                _t = threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t + 1}")
                self.thread_pool.append(_t)
            for _t in self.thread_pool:
                _t.start()

    def get_write_start_frame(self):
        """
        Get Start Frame when start_frame is at its default value
        :return:
        """
        img_list = list()
        for f in os.listdir(self.seq_folder):
            fn, ext = os.path.splitext(f)
            if ext in SupportFormat.img_inputs:
                img_list.append(fn)
        if not len(img_list):
            return 0
        # img_list.sort()
        # last_img = img_list[-1]  # biggest
        return len(img_list)

    def get_frames_cnt(self):
        """
        Get Frames Cnt with EXP
        :return:
        """
        return len(self.img_list) * 2 ** self.exp

    def read_frame(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_LANCZOS4)
        return img

    def write_frame(self, img, path):
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]))
        cv2.imencode(self.output_ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)
        # good

    def nextFrame(self):
        for p in self.img_list:
            img = self.read_frame(p)
            for e in range(2 ** self.exp):
                yield img

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                self.logger.debug(f"{threading.current_thread().name}: get None, break")
                break
            self.write_frame(img_data[1], img_data[0])

    def writeFrame(self, img):
        img_path = os.path.join(self.seq_folder, f"{self.frame_cnt:0>8d}{self.output_ext}")
        img_path = img_path.replace("\\", "/")
        if img is None:
            for t in range(self.thread_cnt):
                self.write_queue.put((img_path, None))
            return
        self.write_queue.put((img_path, img))
        self.frame_cnt += 1
        return

    def close(self):
        for t in range(self.thread_cnt):
            self.write_queue.put(("", None))
        for _t in self.thread_pool:
            while _t.is_alive():
                time.sleep(0.2)
        # if os.path.exists(self.seq_folder):
        #     shutil.rmtree(self.seq_folder)
        return


class SuperResolution:
    """
    超分抽象类
    """

    def __init__(
            self,
            gpuid=0,
            model="models-cunet",
            tta_mode=False,
            num_threads=1,
            scale: float = 2,
            noise=0,
            tilesize=0,
    ):
        self.tilesize = tilesize
        self.noise = noise
        self.scale = scale
        self.num_threads = num_threads
        self.tta_mode = tta_mode
        self.model = model
        self.gpuid = gpuid

    def process(self, im):
        return im

    def svfi_process(self, img):
        """
        SVFI 用于超分的接口
        :param img:
        :return:
        """
        return img


class PathManager:
    """
    路径管理器
    """

    def __init__(self):
        pass


class ArgumentManager:
    """
    For OLS's arguments input management
    """
    app_id = 1692080
    pro_dlc_id = [1718750]

    community_qq = 264023742
    professional_qq = 1054016374

    """Release Version Control"""
    is_steam = True
    is_free = False
    is_release = True
    traceback_limit = 0 if is_release else None
    gui_version = "3.6.12"
    version_tag = f"{gui_version}.beta.1 " \
                  f"{'Professional' if not is_free else 'Community'} - {'Steam' if is_steam else 'Retail'}"
    ols_version = "6.10.13"
    """ 发布前改动以上参数即可 """

    f"""
    Update Log
    - Fallback from 3.6.13 to 3.6.12 to remove SR Pipe
    """

    path_len_limit = 230

    def __init__(self, args: dict):
        self.app_dir = args.get("app_dir", appDir)

        self.config = args.get("config", "")
        self.input = args.get("input", "")
        self.output_dir = args.get("output_dir", "")
        self.task_id = args.get("task_id", "")
        self.gui_inputs = args.get("gui_inputs", "")
        self.input_fps = args.get("input_fps", 0)
        self.target_fps = args.get("target_fps", 0)
        self.output_ext = args.get("output_ext", ".mp4")
        self.is_img_input = args.get("is_img_input", False)
        self.is_img_output = args.get("is_img_output", False)
        self.is_output_only = args.get("is_output_only", True)
        self.is_save_audio = args.get("is_save_audio", True)
        self.input_start_point = args.get("input_start_point", None)
        self.input_end_point = args.get("input_end_point", None)
        if self.input_start_point == "00:00:00":
            self.input_start_point = None
        if self.input_end_point == "00:00:00":
            self.input_end_point = None
        self.output_chunk_cnt = args.get("output_chunk_cnt", 0)
        self.interp_start = args.get("interp_start", 0)
        self.risk_resume_mode = args.get("risk_resume_mode", False)

        self.is_no_scdet = args.get("is_no_scdet", False)
        self.is_scdet_mix = args.get("is_scdet_mix", False)
        self.use_scdet_fixed = args.get("use_scdet_fixed", False)
        self.is_scdet_output = args.get("is_scdet_output", False)
        self.scdet_threshold = args.get("scdet_threshold", 12)
        self.scdet_fixed_max = args.get("scdet_fixed_max", 40)
        self.scdet_flow_cnt = args.get("scdet_flow_cnt", 4)
        self.scdet_mode = args.get("scdet_mode", 0)
        self.remove_dup_mode = args.get("remove_dup_mode", 0)
        self.remove_dup_threshold = args.get("remove_dup_threshold", 0.1)
        self.use_dedup_sobel = args.get("use_dedup_sobel", False)

        self.use_manual_buffer = args.get("use_manual_buffer", False)
        self.manual_buffer_size = args.get("manual_buffer_size", 1)

        self.resize_width = args.get("resize_width", "")
        self.resize_height = args.get("resize_height", "")
        self.resize = args.get("resize", "")
        self.resize_exp = args.get("resize_exp", 1)
        self.crop_width = args.get("crop_width", "")
        self.crop_height = args.get("crop_height", "")
        self.crop = args.get("crop", "")

        self.use_sr = args.get("use_sr", False)
        self.use_sr_algo = args.get("use_sr_algo", "")
        self.use_sr_model = args.get("use_sr_model", "")
        self.use_sr_mode = args.get("use_sr_mode", "")
        self.sr_tilesize = args.get("sr_tilesize", 200)

        self.render_gap = args.get("render_gap", 1000)
        self.use_crf = args.get("use_crf", True)
        self.use_bitrate = args.get("use_bitrate", False)
        self.render_crf = args.get("render_crf", 16)
        self.render_bitrate = args.get("render_bitrate", 90)
        self.render_encoder_preset = args.get("render_encoder_preset", "slow")
        self.render_encoder = args.get("render_encoder", "")
        self.render_hwaccel_mode = args.get("render_hwaccel_mode", "")
        self.render_hwaccel_preset = args.get("render_hwaccel_preset", "")
        self.use_hwaccel_decode = args.get("use_hwaccel_decode", True)
        self.use_manual_encode_thread = args.get("use_manual_encode_thread", False)
        self.render_encode_thread = args.get("render_encode_thread", 16)
        self.is_quick_extract = args.get("is_quick_extract", True)
        self.hdr_mode = args.get("hdr_mode", 0)
        self.render_ffmpeg_customized = args.get("render_ffmpeg_customized", "")
        self.is_no_concat = args.get("is_no_concat", False)
        self.use_fast_denoise = args.get("use_fast_denoise", False)
        self.gif_loop = args.get("gif_loop", True)
        self.is_render_slow_motion = args.get("is_render_slow_motion", False)
        self.render_slow_motion_fps = args.get("render_slow_motion_fps", 0)
        self.use_deinterlace = args.get("use_deinterlace", False)

        self.use_ncnn = args.get("use_ncnn", False)
        self.ncnn_thread = args.get("ncnn_thread", 4)
        self.ncnn_gpu = args.get("ncnn_gpu", 0)
        self.rife_tta_mode = args.get("rife_tta_mode", 0)
        self.rife_tta_iter = args.get("rife_tta_iter", 1)
        self.use_evict_flicker = args.get("use_evict_flicker", False)
        self.use_rife_fp16 = args.get("use_rife_fp16", False)
        self.rife_scale = args.get("rife_scale", 1.0)
        self.rife_model_dir = args.get("rife_model_dir", "")
        self.rife_model = args.get("rife_model", "")
        self.rife_model_name = args.get("rife_model_name", "")
        self.rife_exp = args.get("rife_exp", 1.0)
        self.rife_cuda_cnt = args.get("rife_cuda_cnt", 0)
        self.is_rife_reverse = args.get("is_rife_reverse", False)
        self.use_specific_gpu = args.get("use_specific_gpu", 0)  # !
        self.use_rife_auto_scale = args.get("use_rife_auto_scale", False)
        self.rife_auto_scale_predict_size = args.get("rife_auto_scale_predict_size", 64)
        self.use_rife_forward_ensemble = args.get("use_rife_forward_ensemble", False)
        self.use_rife_multi_cards = args.get("use_rife_multi_cards", False)

        self.debug = args.get("debug", False)
        self.multi_task_rest = args.get("multi_task_rest", False)
        self.multi_task_rest_interval = args.get("multi_task_rest_interval", 1)
        self.after_mission = args.get("after_mission", False)
        self.force_cpu = args.get("force_cpu", False)
        self.expert_mode = args.get("expert_mode", False)
        self.preview_args = args.get("preview_args", False)
        self.is_rude_exit = args.get("is_rude_exit", False)
        self.pos = args.get("pos", "")
        self.size = args.get("size", "")

        """OLS Params"""
        self.concat_only = args.get("concat_only", False)
        self.extract_only = args.get("extract_only", False)
        self.render_only = args.get("render_only", False)
        self.version = args.get("version", "0.0.0 beta")

class VideoFrameInterpolation:
    def __init__(self, __args):
        self.initiated = False
        self.args = {}
        if __args is not None:
            """Update Args"""
            self.args = __args
        else:
            raise NotImplementedError("Args not sent in")

        self.device = None
        self.model = None
        self.model_path = ""

    def initiate_algorithm(self):
        raise NotImplementedError()

    def generate_n_interp(self, img0, img1, n, scale, debug=False):
        raise NotImplementedError()

    def get_auto_scale(self, img1, img2):
        raise NotImplementedError()

    def __make_n_inference(self, img1, img2, scale, n):
        raise NotImplementedError("Abstract")

    def run(self):
        raise NotImplementedError("Abstract")


class Hdr10PlusProcesser:
    def __init__(self, logger: logging, project_dir: str, args: ArgumentManager,
                 interpolation_exp: int, video_info: dict, **kwargs):
        """

        :param logger:
        :param project_dir:
        :param args:
        :param kwargs:
        """
        self.logger = logger
        self.project_dir = project_dir
        self.ARGS = args
        self.interp_exp = interpolation_exp
        self.video_info = video_info
        self.hdr10plus_metadata_4interp = []
        self._initialize()

    def _initialize(self):
        if not len(self.video_info['hdr10plus_metadata']):
            return
        hdr10plus_metadata = self.video_info['hdr10plus_metadata'].copy()
        hdr10plus_metadata = hdr10plus_metadata['SceneInfo']
        hdr10plus_metadata.sort(key=lambda x: int(x['SceneFrameIndex']))
        current_index = -1
        for m in hdr10plus_metadata:
            for j in range(int(self.interp_exp)):
                current_index += 1
                _m = m.copy()
                _m['SceneFrameIndex'] = current_index
                self.hdr10plus_metadata_4interp.append(_m)
        return

    def get_hdr10plus_metadata_at_point(self, start_frame: 0):
        """

        :return: path of metadata json to use immediately
        """
        if not len(self.hdr10plus_metadata_4interp) or start_frame < 0 or start_frame > len(
                self.hdr10plus_metadata_4interp):
            return ""
        if start_frame + self.ARGS.render_gap < len(self.hdr10plus_metadata_4interp):
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:start_frame + self.ARGS.render_gap]
        else:
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:]
        hdr10plus_metadata_path = os.path.join(self.project_dir,
                                               f'hdr10plus_metadata_{start_frame}_{start_frame + self.ARGS.render_gap}.json')
        json.dump(hdr10plus_metadata, open(hdr10plus_metadata_path, 'w'))
        return hdr10plus_metadata_path.replace('/', '\\')


class DoviProcesser:
    def __init__(self, concat_input: str, logger: logging, project_dir: str, args: ArgumentManager,
                 interpolation_exp: int, **kwargs):
        """

        :param concat_input:
        :param logger:
        :param project_dir:
        :param args:
        :param kwargs:
        """
        self.concat_input = concat_input
        self.logger = logger
        self.project_dir = project_dir
        self.ARGS = args
        self.interp_exp = interpolation_exp
        self.ffmpeg = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "ffmpeg.exe"))
        self.ffprobe = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "ffprobe.exe"))
        self.dovi_tool = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "dovi_tool.exe"))
        self.dovi_muxer = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "dovi_muxer.exe"))
        if self.ARGS.ffmpeg == 'ffmpeg':
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
            self.dovi_tool = "dovi_tool"
            self.dovi_muxer = "dovi_muxer"
        self.video_info, self.audio_info = {}, {}
        self.dovi_profile = 8
        self.get_input_info()
        self.concat_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"concat_video.{self.video_info['codec_name']}"))
        self.dv_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"dv_video.{self.video_info['codec_name']}"))
        self.dv_audio_stream = ""
        self.dv_before_rpu = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_before_rpu.rpu"))
        self.rpu_edit_json = os.path.join(self.project_dir, 'rpu_duplicate_edit.json')
        self.dv_after_rpu = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_after_rpu.rpu"))
        self.dv_injected_video_stream = Tools.fillQuotation(
            os.path.join(self.project_dir, f"dv_injected_video.{self.video_info['codec_name']}"))
        self.dv_concat_output_path = Tools.fillQuotation(f'{os.path.splitext(self.concat_input)[0]}_dovi.mp4')

    def get_input_info(self):
        check_command = (f'{self.ffprobe} -v error '
                         f'-show_streams -print_format json '
                         f'{Tools.fillQuotation(self.ARGS.input)}')
        result = check_output(shlex.split(check_command))
        try:
            stream_info = json.loads(result)['streams']  # select first video stream as input
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {result}")
            raise e
        for stream in stream_info:
            if stream['codec_type'] == 'video':
                self.video_info = stream
                break
        for stream in stream_info:
            if stream['codec_type'] == 'audio':
                self.audio_info = stream
                break
        self.logger.info(f"DV Processing [0] - Information gathered, Start Extracting")
        pass

    def run(self):
        try:
            self.split_video2va()
            self.extract_rpu()
            self.modify_rpu()
            self.inject_rpu()
            result = self.mux_va()
            return result
        except Exception:
            self.logger.error("Dovi Conversion Failed")
            raise Exception

    def split_video2va(self):
        audio_map = {'eac3': 'ec3'}
        command_line = (
            f"{self.ffmpeg} -i {Tools.fillQuotation(self.concat_input)} -c:v copy -an -f {self.video_info['codec_name']} {self.concat_video_stream} -y")
        check_output(command_line)
        if len(self.audio_info):
            audio_ext = self.audio_info['codec_name']
            if self.audio_info['codec_name'] in audio_map:
                audio_ext = audio_map[self.audio_info['codec_name']]
            self.dv_audio_stream = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_audio.{audio_ext}"))
            command_line = (
                f"{self.ffmpeg} -i {Tools.fillQuotation(self.ARGS.input)} -c:a copy -vn -f {self.audio_info['codec_name']} {self.dv_audio_stream} -y")
            check_output(command_line)
        self.logger.info(f"DV Processing [1] - Video and Audio track Extracted, start RPU Extracting")

        pass

    def extract_rpu(self):
        command_line = (
            f"{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.ARGS.input)} -c:v copy "
            f'-vbsf {self.video_info["codec_name"]}_mp4toannexb -f {self.video_info["codec_name"]} - | {self.dovi_tool} extract-rpu --rpu-out {self.dv_before_rpu} -')
        check_output(command_line, shell=True)
        self.logger.info(f"DV Processing [2] - Dolby Vision RPU layer extracted, start RPU Modifying")
        pass

    def modify_rpu(self):
        command_line = (
            f"{self.dovi_tool} info -i {self.dv_before_rpu} -f 0")
        rpu_info = check_output(command_line)
        try:
            rpu_info = re.findall('dovi_profile: (.*?),\s.*?offset: (\d+), len: (\d+)', rpu_info.decode())[0]
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {rpu_info}")
            raise e
        self.dovi_profile, dovi_offset, dovi_len = map(lambda x: int(x), rpu_info)
        if 'nb_frames' in self.video_info:
            dovi_len = int(self.video_info['nb_frames'])
        elif 'r_frame_rate' in self.video_info and 'duration' in self.video_info:
            frame_rate = self.video_info['r_frame_rate'].split('/')
            frame_rate = int(frame_rate[0]) / int(frame_rate[1])
            dovi_len = int(frame_rate * float(self.video_info['duration']))

        duplicate_list = []
        for frame in range(dovi_len):
            duplicate_list.append({'source': frame, 'offset': frame, 'length': self.interp_exp - 1})
        edit_dict = {'duplicate': duplicate_list}
        with open(self.rpu_edit_json, 'w') as w:
            json.dump(edit_dict, w)
        command_line = (
            f"{self.dovi_tool} editor -i {self.dv_before_rpu} -j {Tools.fillQuotation(self.rpu_edit_json)} -o {self.dv_after_rpu}")
        check_output(command_line)
        self.logger.info(
            f"DV Processing [3] - RPU layer modified with duplication {self.interp_exp - 1} at length {dovi_len}, start RPU Injecting")

        pass

    def inject_rpu(self):
        command_line = (
            f"{self.dovi_tool} inject-rpu -i {self.concat_video_stream} --rpu-in {self.dv_after_rpu} -o {self.dv_injected_video_stream}")
        check_output(command_line)
        self.logger.info(f"DV Processing [4] - RPU layer Injected to interpolated stream, start muxing")

        pass

    def mux_va(self):
        audio_path = ''
        if len(self.audio_info):
            audio_path = f"-i {Tools.fillQuotation(self.dv_audio_stream)}"
        command_line = f"{self.dovi_muxer} -i {self.dv_injected_video_stream} {audio_path} -o {self.dv_concat_output_path} " \
                       f"--dv-profile {self.dovi_profile} --mpeg4-comp-brand mp42,iso6,isom,msdh,dby1 --overwrite --dv-bl-compatible-id 1"
        check_output(command_line)
        self.logger.info(
            f"DV Processing [5] - interpolated stream muxed to destination: {Tools.get_filename(self.dv_concat_output_path)}")
        self.logger.info(f"DV Processing FINISHED")
        return True


class VideoInfo:
    def __init__(self, file_input: str, logger, project_dir: str, app_dir=None, img_input=False,
                 hdr_mode=False, exp=0, **kwargs):
        self.filepath = file_input
        self.img_input = img_input
        self.hdr_mode = -1
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.hdr10_parser = "hdr10plus_parser"
        self.logger = logger
        self.project_dir = project_dir
        if app_dir is not None:
            self.ffmpeg = Tools.fillQuotation(os.path.join(app_dir, "ffmpeg.exe"))
            self.ffprobe = Tools.fillQuotation(os.path.join(app_dir, "ffprobe.exe"))
            self.hdr10_parser = Tools.fillQuotation(os.path.join(app_dir, "hdr10plus_parser.exe"))
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
            self.hdr10_parser = "hdr10plus_parser"
        self.color_info = dict()
        self.exp = exp
        self.frames_cnt = 0
        self.frames_size = (0, 0)  # width, height
        self.fps = 0
        self.duration = 0
        self.video_info = dict()
        self.hdr10plus_metadata = ""
        self.update_info()

    def update_hdr_mode(self):
        if any([i in str(self.video_info["video_info"]) for i in ['dv_profile', 'DOVI']]):
            # Dolby Vision
            self.hdr_mode = 3
            self.logger.warning("Dolby Vision Content Detected")
            return
        if "color_transfer" not in self.video_info["video_info"]:
            self.logger.warning("Not Find Color Transfer Characteristics")
            self.hdr_mode = 0
            return
        color_trc = self.video_info["video_info"]["color_transfer"]
        if "smpte2084" in color_trc or "bt2020" in color_trc:
            """should be 10bit encoding"""
            self.hdr_mode = 1  # hdr(normal)
            self.logger.warning("HDR Content Detected")
            if any([i in str(self.video_info["video_info"]).lower()]
                   for i in ['mastering-display', "mastering display", "content light level metadata"]):
                self.hdr_mode = 2  # hdr10
                self.logger.warning("HDR10+ Content Detected")
                self.hdr10plus_metadata = os.path.join(self.project_dir, "hdr10plus_metadata.json")
                check_command = (f'{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.filepath)} -c:v copy '
                                 f'-vbsf hevc_mp4toannexb -f hevc - | '
                                 f'{self.hdr10_parser} -o {Tools.fillQuotation(self.hdr10plus_metadata)} -')
                try:
                    check_output(shlex.split(check_command), shell=True)
                except Exception:
                    self.logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))

        elif "arib-std-b67" in color_trc:
            self.hdr_mode = 4  # HLG
            self.logger.warning("HLG Content Detected")
        pass

    def update_frames_info_ffprobe(self):
        check_command = (f'{self.ffprobe} -v error -show_streams -select_streams v:0 -v error '
                         f'-show_entries stream=index,width,height,r_frame_rate,nb_frames,duration,'
                         f'color_primaries,color_range,color_space,color_transfer -print_format json '
                         f'{Tools.fillQuotation(self.filepath)}')
        result = check_output(shlex.split(check_command))
        try:
            video_info = json.loads(result)["streams"][0]  # select first video stream as input
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {result}")
            raise e
        self.video_info = video_info
        self.video_info['video_info'] = video_info
        self.logger.info(f"\nInput Video Info\n{video_info}")
        # update color info
        if "color_range" in video_info:
            self.color_info["-color_range"] = video_info["color_range"]
        if "color_space" in video_info:
            self.color_info["-colorspace"] = video_info["color_space"]
        if "color_transfer" in video_info:
            self.color_info["-color_trc"] = video_info["color_transfer"]
        if "color_primaries" in video_info:
            self.color_info["-color_primaries"] = video_info["color_primaries"]

        self.update_hdr_mode()

        # update frame size info
        if 'width' in video_info and 'height' in video_info:
            self.frames_size = (int(video_info['width']), int(video_info['height']))

        if "r_frame_rate" in video_info:
            fps_info = video_info["r_frame_rate"].split('/')
            self.fps = int(fps_info[0]) / int(fps_info[1])
            self.logger.info(f"Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            self.logger.warning("Auto Find FPS Failed")
            return False

        if "nb_frames" in video_info:
            self.frames_cnt = int(video_info["nb_frames"])
            self.logger.info(f"Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in video_info:
            self.duration = float(video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            self.logger.info(f"Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            self.logger.warning("FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        # if not os.path.isfile(self.filepath):
        #     height, width = 0, 0
        video_input = cv2.VideoCapture(self.filepath)
        if not self.fps:
            self.fps = video_input.get(cv2.CAP_PROP_FPS)
        if not self.frames_cnt:
            self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.duration:
            self.duration = self.frames_cnt / self.fps
        self.frames_size = (
            round(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), round(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def update_info(self):
        if self.img_input:
            if os.path.isfile(self.filepath):
                self.filepath = os.path.dirname(self.filepath)
            seqlist = os.listdir(self.filepath)
            self.frames_cnt = len(seqlist) * 2 ** self.exp
            img = cv2.imdecode(np.fromfile(os.path.join(self.filepath, seqlist[0]), dtype=np.uint8), 1)[:, :,
                  ::-1].copy()
            self.frames_size = (int(img.shape[1]), int(img.shape[0]))
            return
        self.update_frames_info_ffprobe()
        self.update_frames_info_cv2()

    def get_info(self):
        get_dict = {}
        get_dict.update(self.color_info)
        get_dict.update({"video_info": self.video_info})
        get_dict["fps"] = self.fps
        get_dict["size"] = self.frames_size
        get_dict["cnt"] = self.frames_cnt
        get_dict["duration"] = self.duration
        get_dict['hdr_mode'] = self.hdr_mode
        if os.path.exists(self.hdr10plus_metadata):
            hdr10plus_metadata = {}
            try:
                hdr10plus_metadata = json.load(open(self.hdr10plus_metadata, 'r'))
            except json.JSONDecodeError:
                self.logger.error("Unable to Decode HDR10Plus Metadata")
            get_dict['hdr10plus_metadata'] = hdr10plus_metadata
        else:
            get_dict['hdr10plus_metadata'] = {}
        return get_dict


class TransitionDetection_ST:
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=50, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, scdet_output=False):
        """

        :param project_dir: 项目所在文件夹
        :param scene_queue_length:
        :param scdet_threshold:
        :param no_scdet: 无转场检测
        :param use_fixed_scdet: 使用固定转场识别
        :param fixed_max_scdet: 固定转场识别模式下的阈值
        :param scdet_output:
        """
        self.scdet_output = scdet_output
        self.scdet_threshold = scdet_threshold
        self.use_fixed_scdet = use_fixed_scdet
        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.scene_checked_queue = deque(maxlen=self.scene_stack_len // 2)  # 已判断的转场absdiff特判队列
        self.utils = Tools
        self.dead_thres = 60
        self.born_thres = 2
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.scene_dir = os.path.join(project_dir, "scene")
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scene_stack = Queue(maxsize=scene_queue_length)
        self.no_scdet = no_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return sub_array.var() ** 0.65

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}")
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.save_scene(title)

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])), )
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
            cv2.moveWindow(title, 500, 500)
            cv2.resizeWindow(title, 1920, 540)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    def check_scene(self, _img1, _img2, add_diff=False, no_diff=False, use_diff=-1, **kwargs) -> bool:
        """
        Check if current scene is scene
        :param use_diff:
        :param _img2:
        :param _img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回真
        """
        img1 = _img1.copy()
        img2 = _img2.copy()
        self.img1 = img1
        self.img2 = img2

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
                return True

        """检测开头黑场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        # Check really hard scene at the beginning
        if diff > self.dead_thres:
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        """Duplicate Frames Special Judge"""
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        """Judge"""
        return self.__judge_mean(diff)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class TransitionDetection:
    def __init__(self, scene_queue_length, scdet_threshold=50, project_dir="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, remove_dup_mode=0, scdet_output=False, scdet_flow=0,
                 **kwargs):
        """
        转场检测类
        :param scdet_flow: 输入光流模式：0：2D 1：3D
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        :param kwargs:
        """
        self.view = False
        self.utils = Tools
        self.scdet_cnt = 0
        self.scdet_threshold = scdet_threshold
        self.scene_dir = os.path.join(project_dir, "scene")  # 存储转场图片的文件夹路径
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)

        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 3  # 写死判定为非转场的最低阈值

        self.scene_queue_len = scene_queue_length
        if remove_dup_mode in [1, 2]:
            """去除重复帧一拍二或N"""
            self.scene_queue_len = 8  # 写死

        self.flow_queue = deque(maxlen=self.scene_queue_len)  # flow_cnt队列
        self.black_scene_queue = deque(maxlen=self.scene_queue_len)  # 黑场景特判队列
        self.absdiff_queue = deque(maxlen=self.scene_queue_len)  # absdiff队列
        self.scene_stack = Queue(maxsize=self.scene_queue_len)  # 转场识别队列

        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        self.img1 = None
        self.img2 = None
        self.flow_img = None
        self.before_img = None
        if self.use_fixed_scdet:
            self.dead_thres = fixed_max_scdet

        self.scene_output = scdet_output
        if scdet_flow == 0:
            self.scdet_flow = 3
        else:
            self.scdet_flow = 1

        self.now_absdiff = -1
        self.now_vardiff = -1
        self.now_flow_cnt = -1

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.flow_queue))).reshape(-1, 1), np.array(self.flow_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        """
        计算“转场”方差
        :return:
        """
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.flow_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.flow_queue)
        sub_array = np.abs(diff_array - coef_array)
        return sub_array.var() ** 0.65

    def __judge_mean(self, flow_cnt, diff, flow):
        # absdiff_mean = 0
        # if len(self.absdiff_queue) > 1:
        #     self.absdiff_queue.pop()
        #     absdiff_mean = np.mean(self.absdiff_queue)

        var_before = self.__check_var()
        self.flow_queue.append(flow_cnt)
        var_after = self.__check_var()
        self.now_absdiff = diff
        self.now_vardiff = var_after - var_before
        self.now_flow_cnt = flow_cnt
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres and flow_cnt > np.mean(
                self.flow_queue):
            """Detect new scene"""
            self.see_flow(
                f"flow_cnt: {flow_cnt:.3f}, diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, "
                f"cnt: {self.scdet_cnt + 1}", flow)
            self.flow_queue.clear()
            self.scdet_cnt += 1
            self.save_flow()
            return True
        else:
            if diff > self.dead_thres:
                """不漏掉死差转场"""
                self.flow_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                self.save_flow()
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
            self.absdiff_queue.append(diff)
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.see_result(title)

    def see_result(self, title):
        """捕捉转场帧预览"""
        if not self.view:
            return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_flow(self):
        if not self.scene_output:
            return
        try:
            cv2.putText(self.flow_img,
                        f"diff: {self.now_absdiff:.2f}, vardiff: {self.now_vardiff:.2f}, flow: {self.now_flow_cnt:.2f}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.imencode('.png', cv2.cvtColor(self.flow_img, cv2.COLOR_RGB2BGR))[1].tofile(
                os.path.join(self.scene_dir, f"{self.scdet_cnt:08d}.png"))
        except Exception:
            traceback.print_exc()
        pass

    def see_flow(self, title, img):
        """捕捉转场帧光流"""
        if not self.view:
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, img)
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, _img1, _img2, add_diff=False, use_diff=-1.0) -> bool:
        """
                检查当前img1是否是转场
                :param use_diff: 使用已计算出的absdiff
                :param _img2:
                :param _img1:
                :param add_diff: 仅添加absdiff到计算队列中
                :return: 是转场则返回真
                """
        img1 = _img1.copy()
        img2 = _img2.copy()

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            else:
                self.scdet_cnt += 1
                return True

        self.img1 = img1
        self.img2 = img2

        """检测开头转场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.see_result(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            self.flow_img = img1
            self.save_flow()
            return True

        flow_cnt, flow = self.utils.get_norm_img_flow(img1, img2, flow_thres=self.scdet_flow)

        self.absdiff_queue.append(diff)
        self.flow_img = flow

        if len(self.flow_queue) < self.scene_queue_len or add_diff or self.utils.check_pure_img(img1):
            """检测到纯色图片，那么下一帧大概率可以被识别为转场"""
            if flow_cnt > 0:
                self.flow_queue.append(flow_cnt)
            return False

        if flow_cnt == 0:
            return False

        """Judge"""
        return self.__judge_mean(flow_cnt, diff, flow)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class OverTimeReminderBearer:
    reminders = {}

    def generate_reminder(self, *args, **kwargs):
        while True:
            t = random.randrange(100000, 999999)
            if t not in self.reminders:
                reminder = OvertimeReminder(*args, **kwargs)
                self.reminders[t] = reminder
                return t

    def terminate_reminder(self, reminder_id: int):
        if reminder_id not in self.reminders:
            raise threading.ThreadError(f"Do not exist reminder {reminder_id}")
        self.reminders[reminder_id].terminate()

    def terminate_all(self):
        for reminder in self.reminders.values():
            reminder.terminate()


class OvertimeReminder(threading.Thread):
    def __init__(self, interval: int, logger=None, msg_1="Function Type", msg_2="Function Warning", callback=None,
                 *args, **kwargs):
        super().__init__()
        self.logger = logger
        if self.logger is None:
            self.logger = Tools.get_logger("OverTime Reminder", "")
        self.interval = interval
        self.msg_1 = msg_1
        self.msg_2 = msg_2
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.terminated = False
        self.start()

    def run(self):
        start_time = 0
        while start_time < self.interval:
            time.sleep(1)
            if self.terminated:
                return
        if not self.terminated:
            self.logger.warning(f"Function [{self.msg_1}] exceeds {self.interval} seconds, {self.msg_2}")
        if self.callback is not None:
            self.logger.debug(f"OvertimeReminder Callback launch: type {type(self.callback)}")
            self.callback(*self.args, **self.kwargs)
        return

    def terminate(self):
        self.terminated = True


utils_overtime_reminder_bearer = OverTimeReminderBearer()


def overtime_reminder_deco(interval: int, logger=None, msg_1="Function Type", msg_2="Function Warning"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            reminder_id = utils_overtime_reminder_bearer.generate_reminder(interval, logger, msg_1, msg_2)
            result = func(*args, **kwargs)
            utils_overtime_reminder_bearer.terminate_reminder(reminder_id)
            return result

        return wrapper

    return decorator


class SteamUtils:

    def CheckModuleMd5(self):
        """
        Check Integrity of Steam DLLs
        :return:
        """
        steam_api_path = os.path.join(appDir, "steam_api64.dll")
        steam_py_path = os.path.join(appDir, "steamworks", "SteamworksPy64.dll")

    def CheckSteamAuth(self):
        if self.is_steam:
            return 0
        steam_64id = self.steamworks.Users.GetSteamID()
        valid_response = self.steamworks.Users.GetAuthSessionTicket()
        self.logger.info(f'Steam User Logged on as {steam_64id}, auth: {valid_response}')
        return valid_response

    def CheckProDLC(self, dlc_id: int) -> bool:
        """

        :param dlc_id: DLC for SVFI, start from 0
        0: Pro
        :return:
        """
        if not self.is_steam:
            return False
        purchase_pro = self.steamworks.Apps.IsDLCInstalled(ArgumentManager.pro_dlc_id[dlc_id])
        self.logger.info(f'Steam User Purchase Pro DLC Status: {purchase_pro}')
        return purchase_pro

    def __init__(self, is_steam, logger=None):
        """
        Whether use steam for validation
        :param is_steam:
        """
        original_cwd = os.getcwd()
        self.is_steam = is_steam
        if logger is None:
            self.logger = Tools().get_logger(__file__, "")
        else:
            self.logger = logger
        self.steamworks = None
        self.steam_valid = True
        self.steam_error = ""
        if self.is_steam:
            self.steamworks = STEAMWORKS(ArgumentManager.app_id)
            self.steamworks.initialize()  # This method has to be called in order for the wrapper to become functional!

            if self.steamworks.UserStats.RequestCurrentStats() == True:
                self.logger.info('Steam Stats successfully retrieved!')
            else:
                self.steam_valid = False
                self.steam_error = GenericSteamException('Failed to get Stats Error, Please Make Sure Steam is On')
                self.logger.error('Failed to get stats. Shutting down.')
        os.chdir(original_cwd)

    def GetStat(self, key: str, key_type: type):
        if not self.is_steam:
            return
        if key_type is int:
            return self.steamworks.UserStats.GetStatInt(key)
        elif key_type is float:
            return self.steamworks.UserStats.GetStatFloat(key)

    def GetAchv(self, key: str):
        if not self.is_steam:
            return False
        return self.steamworks.UserStats.GetAchievement(key)

    def SetStat(self, key: str, value):
        if not self.is_steam:
            return False
        return self.steamworks.UserStats.SetStat(key, value)

    def SetAchv(self, key: str, clear=False):
        if not self.is_steam:
            return False
        if clear:
            return self.steamworks.UserStats.ClearAchievement(key)
        return self.steamworks.UserStats.SetAchievement(key)

    def Store(self):
        if not self.is_steam:
            return False
        return self.steamworks.UserStats.StoreStats()


class EULAWriter:
    eula_hi = """
    EULA
    
    重要须知——请仔细阅读：请确保仔细阅读并理解《最终用户许可协议》（简称“协议”）中描述的所有权利与限制。
    
    协议
    本协议是您与SDT Core及其附属公司（简称“公司”）之间达成的协议。仅在您接受本协议中包含的所有条件的情况下，您方可使用软件及任何附属印刷材料。
    安装或使用软件即表明，您同意接受本《协议》各项条款的约束。如果您不同意本《协议》中的条款：(i)请勿安装软件, (ii)如果您已经购买软件，请立即凭购买凭证将其退回购买处，并获得退款。
    在您安装软件时，会被要求预览并通过点击“我接受”按钮决定接受或不接受本《协议》的所有条款。点击“我接受”按钮，即表明您承认已经阅读过本《协议》，并且理解并同意受其条款与条件的约束。
    版权
    软件受版权法、国际协约条例以及其他知识产权法和条例的保护。软件（包括但不限于软件中含有的任何图片、照片、动画、视频、音乐、文字和小型应用程序）及其附属于软件的任何印刷材料的版权均由公司及其许可者拥有。
    
    许可证的授予
    软件的授权与使用须遵从本《协议》。公司授予您有限的、个人的、非独占的许可证，允许您使用软件，并且以将其安装在您的手机上为唯一目的。公司保留一切未在本《协议》中授予您的权利。
    
    授权使用
    1. 如果软件配置为在一个硬盘驱动器上运行，您可以将软件安装在单一电脑上，以便在您的手机上安装和使用它。
    2. 您可以制作和保留软件的一个副本用于备份和存档，条件是软件及副本归属于您。
    3. 您可以将您在本《协议》项下的所有权利永久转让，转让的条件是您不得保留副本，转让软件（包括全部组件、媒体、印刷材料及任何升级版本），并且受让人接受本《协议》的各项条款。
    
    限制
    1. 您不得删除或掩盖软件或附属印刷材料注明的版权、商标或其他所有权。
    2. 您不得对软件进行反编译、修改、逆向工程、反汇编或重制。
    3. 您不得复制、租赁、发布、散布或公开展示软件，不得制作软件的衍生产品（除非编辑器和本协议最终用户变更部分或其他附属于软件的文件明确许可），或是以商业目的对软件进行开发。
    4. 您不得通过电子方式或网络将软件从一台电脑、控制台或其他平台传送到另一个上。
    5. 您不得将软件的备份或存档副本用作其他用途，只可在原始副本被损坏或残缺的情况下，用其替换原始副本。
    6. 您不得将软件的输出结果用于商业用途
    
    试用版本
    如果提供给您的软件为试用版，其使用期限或使用数量有限制，您同意在试用期结束后停止使用软件。您知晓并同意软件可能包含用于避免您突破这些限制的代码，并且这些代码会在您删除软件后仍保留在您的电脑上，避免您下载其他副本并重复利用试用期。
    
    编辑器和最终用户变更
    如果软件允许您进行修改或创建新内容（“编辑器”），您可以使用该编辑器修改或优化软件，包括创建新内容（统称“变更”），但必须遵守下列限制。您的变更(i)必须符合已注册的完整版软件；(ii)不得对执行文件进行改动；(iii)不得包含任何诽谤、中伤、违法、损害他人或公众利益的内容；(iv)不得包含任何商标、著作权保护内容或第三方的所有权内容；(v)不得用作商业目的，包括但不限于，出售变更内容、按次计费或分时服务。
    
    终止
    本协议在终止前都是有效的。您可以随时卸载软件来终止该协议。如果您违反了协议的任何条款或条件，本协议将自动终止，恕不另行通知。本协议中涉及到的保证、责任限制和损失赔偿的部分在协议终止后仍然有效。
    
    有限保修及免责条款
    您知道并同意因使用该软件及其记录该软件的媒体所产生的风险由您自行承担。该软件和媒体“照原样”发布。除非有适用法律规定，本公司向此产品的原始购买人保证，在正常使用的情况，该软件媒体存储介质在30天内（自购买之日算起）无缺陷。对于因意外、滥用、疏忽或误用引起的缺陷，该保证无效。如果软件没有达到保证要求，您可能会单方面获得补偿，如果您退回有缺陷的软件，您可以免费获得替换产品。本公司不保证该软件及其操作和功能达到您的要求，也不保证软件的使用不会出现中断或错误。
    在适用法律许可的最大范围下，除了上述的明确保证之外，本公司不做其他任何保证，包括但不限于暗含性的适销保证、特殊用途保证及非侵权保证。除了上述的明确保证之外，本公司不对软件使用和软件使用结果在正确性、准确性、可靠性、通用性和其他方面做出保证、担保或陈述。部分司法管辖区不允许排除或限制暗含性保证，因此上面的例外和限制情况可能对您不适用。
    
    责任范围
    在任何情况下，本公司及其员工和授权商都不对任何由软件使用或无法使用软件而引起的任何附带、间接、特殊、偶然或惩罚性伤害以及其他伤害（包括但不限于对人身或财产的伤害，利益损失，运营中断，商业信息丢失，隐私侵犯，履行职责失败及疏忽）负责，即使公司或公司授权代表已知悉了存在这种伤害的可能性。部分司法管辖区不允许排除附带或间接伤害，因此，上述例外情况可能对您不适用。
    
    在任何情况下，公司承担的和软件伤害相关的费用都不超过您对该软件实际支付的数额。
    
    其他
    如果发现此最终用户许可协议的任意条款或规定违法、无效或因某些原因无法强制执行，该条款和部分将被自动舍弃，不会影响本协议其余规定的有效性和可执行性。
    本协议包含软您和本软件公司之间的所有协议及其使用方法。
    
    eula = true
    """

    def __init__(self):
        self.eula_dir = os.path.join(appDir, 'train_log')
        os.makedirs(self.eula_dir, exist_ok=True)
        self.eula_path = os.path.join(self.eula_dir, 'md5.svfi')

    def boom(self):
        with open(self.eula_path, 'w', encoding='utf-8') as w:
            w.write(self.eula_hi)


if __name__ == "__main__":
    # u = Tools()
    # cp = DefaultConfigParser(allow_no_value=True)
    # cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
    #         encoding='utf-8')
    # print(cp.get("General", "UseCUDAButton=true", 6))
    # print(u.clean_parsed_config(dict(cp.items("General"))))
    # dm = DoviMaker(r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_71fps_[S-0.5]_[offical_3.8]_963577.mp4", Tools.get_logger('', ''),
    #                r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_ec4c18_963577",
    #                ArgumentManager(
    #                    {'ffmpeg': r'D:\60-fps-Project\ffmpeg',
    #                     'input': r"E:\Library\Downloads\Video\dolby vision-blocks.mp4"}),
    #                int(72 / 24),
    #                )
    # dm.run()
    print(Tools.check_non_ascii(".fdassda f。"))
    pass


