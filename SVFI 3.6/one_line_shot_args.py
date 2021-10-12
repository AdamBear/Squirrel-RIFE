# coding: utf-8
import argparse
import sys

import tqdm

from Utils.utils import *
from skvideo.io import FFmpegWriter, FFmpegReader, EnccWriter, SVTWriter
from steamworks.exceptions import *

try:
    _steamworks = STEAMWORKS(ArgumentManager.app_id)
except:
    pass

print(f"INFO - ONE LINE SHOT ARGS {ArgumentManager.ols_version} {datetime.date.today()}")
# TODO Fix up SVT-HEVC

"""设置环境路径"""
os.chdir(appDir)
sys.path.append(appDir)

"""输入命令行参数"""
parser = argparse.ArgumentParser(prog="#### SVFI CLI tool by Jeanna ####",
                                 description='Interpolation for long video/imgs footage')
basic_parser = parser.add_argument_group(title="Basic Settings, Necessary")
basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                          help="原视频/图片序列文件夹路径")
basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
basic_parser.add_argument("-t", '--task-id', dest='task_id', type=str, required=True, help="任务id")
basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')
basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true', help='只执行拆帧操作')
basic_parser.add_argument('--render-only', dest='render_only', action='store_true', help='只执行渲染操作')

args_read = parser.parse_args()
cp = DefaultConfigParser(allow_no_value=True)  # 把SVFI GUI传来的参数格式化
cp.read(args_read.config, encoding='utf-8')
cp_items = dict(cp.items("General"))
args = Tools.clean_parsed_config(cp_items)
args.update(vars(args_read))  # update -i -o -c，将命令行参数更新到config生成的字典
ARGS = ArgumentManager(args)

"""设置可见的gpu"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if int(ARGS.rife_cuda_cnt) != 0 and ARGS.use_rife_multi_cards:
    cuda_devices = [str(i) for i in range(ARGS.rife_cuda_cnt)]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(cuda_devices)}"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{ARGS.use_specific_gpu}"

"""强制使用CPU"""
if ARGS.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f""

"""Set Global Logger"""
logger = Tools.get_logger('TMP', '')


class InterpWorkFlow:
    # @profile
    def __init__(self, __args: ArgumentManager, **kwargs):
        global logger
        self.ARGS = __args

        """EULA"""
        self.eula = EULAWriter()
        self.eula.boom()

        """获得补帧输出路径"""
        if not len(self.ARGS.output_dir):
            """未填写输出文件夹"""
            self.ARGS.output_dir = os.path.dirname(self.ARGS.input)
        if os.path.isfile(self.ARGS.output_dir):
            self.ARGS.output_dir = os.path.dirname(self.ARGS.output_dir)
        self.project_name = f"{Tools.get_filename(self.ARGS.input)}_{self.ARGS.task_id}"
        self.project_dir = os.path.join(self.ARGS.output_dir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        sys.path.append(self.project_dir)

        """Set Logger"""
        logger = Tools.get_logger("CLI", self.project_dir, debug=self.ARGS.debug)
        logger.info(f"Initial New Interpolation Project: project_dir: %s, INPUT_FILEPATH: %s", self.project_dir,
                    self.ARGS.input)

        """Steam Validation"""
        self.STEAM = SteamUtils(self.ARGS.is_steam, logger=logger)

        """Set FFmpeg"""
        self.ffmpeg = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "ffmpeg.exe"))
        self.ffplay = Tools.fillQuotation(os.path.join(self.ARGS.app_dir, "ffplay.exe"))
        if not os.path.exists(os.path.join(self.ARGS.app_dir, "ffmpeg.exe")):
            self.ffmpeg = "ffmpeg"
            logger.warning("Not find selected ffmpeg, use default")

        """Set input output and initiate environment"""
        self.input = self.ARGS.input
        self.output = self.ARGS.output_dir
        if self.ARGS.is_img_output:
            self.output = os.path.join(self.output, self.project_name)
            os.makedirs(self.output, exist_ok=True)
        if not os.path.isfile(self.input):
            self.ARGS.is_img_input = True

        """Get input's info"""
        self.video_info_instance = VideoInfo(file_input=self.input, logger=logger, project_dir=self.project_dir,
                                             app_dir=self.ARGS.app_dir, img_input=self.ARGS.is_img_input,
                                             hdr_mode=self.ARGS.hdr_mode, exp=self.ARGS.rife_exp)
        self.video_info = self.video_info_instance.get_info()
        if self.ARGS.hdr_mode == 0:  # Auto
            logger.info(f"Auto HDR Mode, Set HDR mode to {self.video_info['hdr_mode']}")
            self.hdr_check_status = self.video_info['hdr_mode']
            # no hdr at -1, 0 checked and None, 1 hdr, 2 hdr10, 3 DV, 4 HLG
            # hdr_check_status indicates the final process mode for (hdr) input
        else:
            self.hdr_check_status = self.ARGS.hdr_mode

        """Set input and target(output) fps"""
        if not self.ARGS.is_img_input:  # 输入不是文件夹，使用检测到的帧率
            self.input_fps = self.video_info["fps"]
        elif self.ARGS.input_fps:
            self.input_fps = self.ARGS.input_fps
        else:  # 用户有毒，未发现有效的输入帧率，用检测到的帧率
            if self.video_info["fps"] is None or not self.video_info["fps"]:
                raise OSError("Not Find FPS, Input File is not valid")
            self.input_fps = self.video_info["fps"]

        if self.ARGS.is_img_input:
            self.target_fps = self.ARGS.target_fps
            self.ARGS.is_save_audio = False
            # but assigned output fps will be not altered
        else:
            if self.ARGS.target_fps:
                self.target_fps = self.ARGS.target_fps
            else:
                self.target_fps = (2 ** self.ARGS.rife_exp) * self.input_fps  # default

        """Set interpolation exp related to hdr mode"""
        self.interp_exp = self.target_fps / self.input_fps
        if self.hdr_check_status == 3 or (self.hdr_check_status == 2 and len(self.video_info['hdr10plus_metadata'])):
            """DoVi or Valid HDR10 Metadata Dected"""
            self.interp_exp = int(math.ceil(self.target_fps / self.input_fps))
            self.target_fps = self.interp_exp * self.input_fps
        self.hdr10_metadata_processer = Hdr10PlusProcesser(logger, self.project_dir, self.ARGS,
                                                           self.interp_exp, self.video_info)

        """Update All Frames Count"""
        self.max_frame_cnt = 10 ** 10
        self.all_frames_cnt = abs(int(self.video_info["duration"] * self.target_fps))
        if self.all_frames_cnt > self.max_frame_cnt:
            raise OSError(f"SVFI can't afford input exceeding {self.max_frame_cnt} frames")

        """Set Cropping Parameters"""
        self.crop_param = [0, 0]  # crop parameter, 裁切参数
        crop_param = self.ARGS.crop.replace("：", ":")
        if crop_param not in ["", "0", None]:
            width_black, height_black = crop_param.split(":")
            width_black = int(width_black)
            height_black = int(height_black)
            self.crop_param = [width_black, height_black]
            logger.info(f"Update Crop Parameters to {self.crop_param}")

        """Check Initiation Info"""
        logger.info(
            f"Check Interpolation Source, FPS: {self.input_fps}, TARGET FPS: {self.target_fps}, "
            f"FRAMES_CNT: {self.all_frames_cnt}, EXP: {self.ARGS.rife_exp}")

        """Set RIFE Core"""
        self.vfi_core = VideoFrameInterpolation(self.ARGS)  # 用于补帧的模块

        """Guess Memory and Fix Resolution"""
        if self.ARGS.use_manual_buffer:
            # 手动指定内存占用量
            free_mem = self.ARGS.manual_buffer_size * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        if self.ARGS.resize_width != 0 and self.ARGS.resize_height != 0:
            """规整化输出输入分辨率"""
            if self.ARGS.resize_width % 2 != 0:
                self.ARGS.resize_width += 1
            if self.ARGS.resize_height % 2 != 0:
                self.ARGS.resize_height += 1
            self.frames_queue_len = round(free_mem / (sys.getsizeof(
                np.random.rand(3, round(self.ARGS.resize_width),
                               round(self.ARGS.resize_height))) / 1024 / 1024))
        else:
            self.frames_queue_len = round(free_mem / (sys.getsizeof(
                np.random.rand(3, round(self.video_info["size"][0]),
                               round(self.video_info["size"][1]))) / 1024 / 1024))
        if not self.ARGS.use_manual_buffer:
            self.frames_queue_len = int(max(10.0, self.frames_queue_len))
        logger.info(f"Buffer Size to {self.frames_queue_len}")

        """Set Queues"""
        self.frames_output = Queue(maxsize=self.frames_queue_len)  # 补出来的帧序列队列（消费者）
        self.rife_task_queue = Queue(maxsize=self.frames_queue_len)  # 补帧任务队列（生产者）
        self.rife_thread = None  # 帧插值预处理线程（生产者）
        self.rife_work_event = threading.Event()
        self.rife_work_event.clear()

        """Set Render Parameters"""
        self.frame_reader = None  # 读帧的迭代器／帧生成器
        self.render_gap = self.ARGS.render_gap  # 每个chunk的帧数
        self.render_thread = None  # 帧渲染器
        self.task_info = {"chunk_cnt": 0, "render": 0, "now_frame": 0}  # 有关渲染的实时信息

        """Set Super Resolution"""
        self.sr_module = SuperResolution()  # 超分类
        if self.ARGS.use_sr:
            try:
                input_resolution = self.video_info["size"][0] * self.video_info["size"][1]
                # for img input, video info is updated to first img input
                output_resolution = self.ARGS.resize_width * self.ARGS.resize_height
                # for img output, if output_resolution not assigned(0,0), resolution_rate should be 0
                resolution_rate = output_resolution / input_resolution
                sr_scale = 0
                if input_resolution and resolution_rate > 1:
                    sr_scale = int(math.ceil(resolution_rate))
                if self.ARGS.resize_exp > 1:
                    """Compulsorily assign scale = resize_exp, could be img input"""
                    sr_scale = self.ARGS.resize_exp
                    # eventual output resolution will still be affected by assigned output resolution (if not 0,0)
                if sr_scale > 1:
                    resize_param = (self.ARGS.resize_width, self.ARGS.resize_height)
                    if self.ARGS.use_sr_algo == "waifu2x":
                        import Utils.SuperResolutionModule
                        self.sr_module = Utils.SuperResolutionModule.SvfiWaifu(model=self.ARGS.use_sr_model,
                                                                               scale=sr_scale,
                                                                               num_threads=self.ARGS.ncnn_thread,
                                                                               resize=resize_param)
                    elif self.ARGS.use_sr_algo == "realSR":
                        import Utils.SuperResolutionModule
                        self.sr_module = Utils.SuperResolutionModule.SvfiRealSR(model=self.ARGS.use_sr_model,
                                                                                scale=sr_scale,
                                                                                resize=resize_param)
                    elif self.ARGS.use_sr_algo == "realESR":
                        import Utils.RealESRModule
                        self.sr_module = Utils.RealESRModule.SvfiRealESR(model=self.ARGS.use_sr_model,
                                                                         gpu_id=self.ARGS.use_specific_gpu,
                                                                         scale=sr_scale, tile=self.ARGS.sr_tilesize,
                                                                         half=self.ARGS.use_rife_fp16,
                                                                         resize=resize_param)
                    logger.info(
                        f"Load AI SR at {self.ARGS.use_sr_algo}, {self.ARGS.use_sr_model}, "
                        f"scale = {sr_scale}, resize = {resize_param}")
                else:
                    self.ARGS.use_sr = False
                    logger.warning("Abort to load AI SR since Resolution Rate <= 1")
            except ImportError:
                logger.error(
                    f"Import SR Module failed\n{traceback.format_exc(limit=ArgumentManager.traceback_limit)}")

        """Scene Detection"""
        if self.ARGS.scdet_mode == 0:
            """Old Mode"""
        self.scene_detection = TransitionDetection_ST(self.project_dir, int(0.5 * self.input_fps),
                                                      scdet_threshold=self.ARGS.scdet_threshold,
                                                      no_scdet=self.ARGS.is_no_scdet,
                                                      use_fixed_scdet=self.ARGS.use_scdet_fixed,
                                                      fixed_max_scdet=self.ARGS.scdet_fixed_max,
                                                      scdet_output=self.ARGS.is_scdet_output)
        """Duplicate Frames Removal"""
        self.dup_skip_limit = int(0.5 * self.input_fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环

        """Main Thread Lock"""
        self.main_event = threading.Event()
        self.render_lock = threading.Event()  # 渲染锁，没有用
        self.main_event.set()

        """Set output's color info"""
        self.color_info = {}
        for k in self.video_info:
            if k.startswith("-"):
                self.color_info[k] = self.video_info[k]

        """fix extension"""
        self.input_ext = os.path.splitext(self.input)[1] if os.path.isfile(self.input) else ""
        self.input_ext = self.input_ext.lower()
        self.output_ext = "." + self.ARGS.output_ext
        if "ProRes" in self.ARGS.render_encoder and not self.ARGS.is_img_output:
            self.output_ext = ".mov"

        self.reminder_bearer = OverTimeReminderBearer()
        self.main_error = None

    def generate_frame_reader(self, start_frame=-1, frame_check=False):
        """
        输入帧迭代器
        :param frame_check:
        :param start_frame:
        :return:
        """
        """If input is sequence of frames"""
        if self.ARGS.is_img_input:
            img_io = ImgSeqIO(folder=self.input, is_read=True,
                              start_frame=self.ARGS.interp_start, logger=logger,
                              output_ext=self.ARGS.output_ext, exp=self.ARGS.rife_exp,
                              resize=(self.ARGS.resize_width, self.ARGS.resize_height),
                              is_esr=self.ARGS.use_sr_algo == "realESR")
            self.all_frames_cnt = img_io.get_frames_cnt()
            logger.info(f"Img Input, update frames count to {self.all_frames_cnt}")
            return img_io

        """If input is a video"""
        input_dict = {"-vsync": "0", }
        if self.ARGS.use_hwaccel_decode:
            input_dict.update({"-hwaccel": "auto"})

        if self.ARGS.input_start_point or self.ARGS.input_end_point:
            """任意时段任务"""
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime("00:00:00", time_fmt)
            end_point = datetime.datetime.strptime("00:00:00", time_fmt)
            if self.ARGS.input_start_point is not None:
                start_point = datetime.datetime.strptime(self.ARGS.input_start_point, time_fmt) - start_point
                input_dict.update({"-ss": self.ARGS.input_start_point})
            else:
                start_point = start_point - start_point
            if self.ARGS.input_end_point is not None:
                end_point = datetime.datetime.strptime(self.ARGS.input_end_point, time_fmt) - end_point
                input_dict.update({"-to": self.ARGS.input_end_point})
            elif self.video_info['duration']:
                # no need to care about img input
                end_point = datetime.datetime.fromtimestamp(
                    self.video_info['duration']) - datetime.datetime.fromtimestamp(0.0)
            else:
                end_point = end_point - end_point

            if end_point > start_point:
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.target_fps
                self.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                logger.info(
                    f"Update Input Range: in {self.ARGS.input_start_point} -> out {self.ARGS.input_end_point}, all_frames_cnt -> {self.all_frames_cnt}")
            else:
                if '-ss' in input_dict:
                    input_dict.pop('-ss')
                if '-to' in input_dict:
                    input_dict.pop('-to')
                logger.warning(
                    f"Invalid Input Section, change to original course")
        else:
            logger.info(f"Input Time Section is original course")

        output_dict = {
            "-vframes": str(10 ** 10), }  # use read frames cnt to avoid ffprobe, fuck
        # debug
        # output_dict = {}
        output_dict.update(self.color_info)

        if frame_check:
            """用以一拍二一拍N除重模式的预处理"""
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": "300x300"})
        elif self.ARGS.resize_height and self.ARGS.resize_width and not self.ARGS.use_sr:
            h, w = self.video_info["size"][1], self.video_info["size"][0]
            if h != self.ARGS.resize_height or w != self.ARGS.resize_width:
                output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                    "-s": f"{self.ARGS.resize_width}x{self.ARGS.resize_height}"})
        vf_args = f"copy"
        if self.ARGS.use_deinterlace:
            vf_args += f",yadif=parity=auto"
        if start_frame not in [-1, 0]:
            # not start from the beginning
            if self.ARGS.risk_resume_mode:
                """Quick Locate"""
                input_dict.update({"-ss": f"{start_frame / self.target_fps:.3f}"})
            else:
                output_dict.update({"-ss": f"{start_frame / self.target_fps:.3f}"})

        """Quick Extraction"""
        if not self.ARGS.is_quick_extract:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"

        vf_args += f",minterpolate=fps={self.target_fps}:mi_mode=dup"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.input, inputdict=input_dict, outputdict=output_dict)

    def generate_frame_renderer(self, output_path, start_frame=0):
        """
        渲染帧
        :param start_frame: for IMG IO, select start_frame to generate IO instance
        :param output_path:
        :return:
        """
        hdr10plus_metadata = self.hdr10_metadata_processer.get_hdr10plus_metadata_at_point(start_frame)
        params_libx265s = {
            "fast": "high-tier=0:ref=2:rd=1:ctu=32:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=1:"
                    "rdoq-level=0:me=2:subme=3:merange=25:weightb=1:strong-intra-smoothing=0:open-gop=0:keyint=250:"
                    "min-keyint=1:rc-lookahead=25:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:"
                    "crqpoffs=-2:qcomp=0.65:sao=0:repeat-headers=1",
            "8bit": "high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:"
                    "strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:"
                    "rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:"
                    "qcomp=0.65:sao=0",
            "10bit": "high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:"
                     "strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:"
                     "rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:"
                     "sao=0",
            "hdr10": 'high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:'
                     'strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:'
                     'rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:'
                     'sao=0:'
                     'range=limited:colorprim=9:transfer=16:colormatrix=9:'
                     'master-display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)":'
                     'max-cll="1000,100":hdr10-opt=1:repeat-headers=1',
            "hdr10+": 'high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:'
                      'strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:'
                      'rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:'
                      'sao=0:'
                      'range=limited:colorprim=9:transfer=16:colormatrix=9:'
                      'master-display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)":'
                      f'max-cll="1000,100":dhdr10-info="{hdr10plus_metadata}"'
        }

        params_libx264s = {
            "fast": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=4:"
                    "rc-lookahead=30:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:me=hex:merange=16:"
                    "subme=7:psy-rd='1:0.1':mixed-refs=1:trellis=1",
            "8bit": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:"
                    "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                    "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0",
            "10bit": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:"
                     "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                     "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0",
            "hdr10": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:"
                     "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                     "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0:"
                     "range=tv:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                     "mastering-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)':"
                     "cll='1000,100'"
        }

        def HDR_auto_modify_params():
            if self.ARGS.is_img_input or self.hdr_check_status == 1:  # img input or ordinary hdr
                return

            if self.hdr_check_status == 2:
                """HDR10"""
                self.ARGS.render_hwaccel_mode = "CPU"
                if "H265" in self.ARGS.render_encoder:
                    self.ARGS.render_encoder = "H265, 10bit"
                elif "H264" in self.ARGS.render_encoder:
                    self.ARGS.render_encoder = "H264, 10bit"
                self.ARGS.render_encoder_preset = "medium"
            elif self.hdr_check_status == 4:
                """HLG"""
                self.ARGS.render_encoder = "H265, 10bit"
                self.ARGS.render_hwaccel_mode = "CPU"
                self.ARGS.render_encoder_preset = "medium"

        """If output is sequence of frames"""
        if self.ARGS.is_img_output:
            img_io = ImgSeqIO(folder=self.project_dir, is_read=False,
                              start_frame=-1, logger=logger,
                              output_ext=self.ARGS.output_ext, )
            return img_io

        """HDR Check"""
        if self.ARGS.hdr_mode == 0:  # Auto
            HDR_auto_modify_params()

        """Output Video"""
        input_dict = {"-vsync": "cfr"}

        output_dict = {"-r": f"{self.target_fps}", "-preset": self.ARGS.render_encoder_preset,
                       "-metadata": f'title="Powered By SVFI {self.ARGS.version}"'}

        output_dict.update(self.color_info)

        if not self.ARGS.is_img_input:
            input_dict.update({"-r": f"{self.target_fps}"})
        else:
            """Img Input"""
            input_dict.update({"-r": f"{self.input_fps * 2 ** self.ARGS.rife_exp}"})

        """Slow motion design"""
        if self.ARGS.is_render_slow_motion:
            if self.ARGS.render_slow_motion_fps:
                input_dict.update({"-r": f"{self.ARGS.render_slow_motion_fps}"})
            else:
                input_dict.update({"-r": f"{self.target_fps}"})
            output_dict.pop("-r")

        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        if self.ARGS.use_sr and self.ARGS.resize_height and self.ARGS.resize_width:
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": f"{self.ARGS.resize_width}x{self.ARGS.resize_height}"})

        """Assign Render Codec"""
        """CRF / Bitrate Control"""
        if self.ARGS.render_hwaccel_mode == "CPU":
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx264", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "high",
                                        "-x264-params": params_libx264s["8bit"]})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "high10",
                                        "-x264-params": params_libx264s["10bit"]})
                if 'fast' in self.ARGS.render_encoder_preset:
                    output_dict.update({"-x264-params": params_libx264s["fast"]})
                if self.hdr_check_status == 2:
                    """HDR10"""
                    output_dict.update({"-x264-params": params_libx264s["hdr10"]})
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx265", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main",
                                        "-x265-params": params_libx265s["8bit"]})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "main10",
                                        "-x265-params": params_libx265s["10bit"]})
                if 'fast' in self.ARGS.render_encoder_preset:
                    output_dict.update({"-x265-params": params_libx265s["fast"]})
                if self.hdr_check_status == 2:
                    """HDR10"""
                    output_dict.update({"-x265-params": params_libx265s["hdr10"]})
                    if os.path.exists(hdr10plus_metadata):
                        output_dict.update({"-x265-params": params_libx265s["hdr10+"]})
            else:
                """ProRes"""
                if "-preset" in output_dict:
                    output_dict.pop("-preset")
                output_dict.update({"-c:v": "prores_ks", "-profile:v": self.ARGS.render_encoder_preset, })
                if "422" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv422p10le"})
                else:
                    output_dict.update({"-pix_fmt": "yuv444p10le"})

        elif self.ARGS.render_hwaccel_mode == "NVENC":
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({f"-g": f"{int(self.target_fps * 3)}", "-c:v": "h264_nvenc", "-rc:v": "vbr_hq", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq",
                                    f"-g": f"{int(self.target_fps * 3)}", })

            if self.ARGS.render_encoder_preset != "loseless":
                hwacccel_preset = self.ARGS.render_hwaccel_preset
                if hwacccel_preset != "None":
                    output_dict.update({"-i_qfactor": "0.71", "-b_qfactor": "1.3", "-keyint_min": "1",
                                        f"-rc-lookahead": "120", "-forced-idr": "1", "-nonref_p": "1",
                                        "-strict_gop": "1", })
                    if hwacccel_preset == "5th":
                        output_dict.update({"-bf": "0"})
                    elif hwacccel_preset == "6th":
                        output_dict.update({"-bf": "0", "-weighted_pred": "1"})
                    elif hwacccel_preset == "7th+":
                        output_dict.update({"-bf": "4", "-temporal-aq": "1", "-b_ref_mode": "2"})
            else:
                output_dict.update({"-preset": "10", })

        elif self.ARGS.render_hwaccel_mode == "NVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "NVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                # "--chroma-qp-offset": "-2",
                "--lookahead": "16",
                "--gop-len": "250",
                "-b": "4",
                "--ref": "8",
                "--aq": "",
                "--aq-temporal": "",
                "--bref-mode": "middle"}
            if '-color_range' in output_dict:
                _output_dict.update({"--colorrange": output_dict["-color_range"]})
            if '-colorspace' in output_dict:
                _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            if '-color_trc' in output_dict:
                _output_dict.update({"--transfer": output_dict["-color_trc"]})
            if '-color_primaries' in output_dict:
                _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encoder:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high10" if "10bit" in self.ARGS.render_encoder else "high", })
            elif "H265" in self.ARGS.render_encoder:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "-b": "5"})

            if self.hdr_check_status == 2:
                """HDR10"""
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10",
                                     "--tier": "main", "-b": "5",
                                     "--max-cll": "1000,100",
                                     "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"})
                if os.path.exists(hdr10plus_metadata):
                    _output_dict.update({"--dhdr10-info": hdr10plus_metadata})
            else:
                if self.ARGS.render_encoder_preset != "loseless":
                    _output_dict.update({"--preset": self.ARGS.render_encoder_preset})
                else:
                    _output_dict.update({"--lossless": "", "--preset": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_hwaccel_mode == "QSVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "QSVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                "--fallback-rc": "", "--la-depth": "50", "--la-quality": "slow", "--extbrc": "", "--mbbrc": "",
                "--i-adapt": "",
                "--b-adapt": "", "--gop-len": "250", "-b": "6", "--ref": "8", "--b-pyramid": "", "--weightb": "",
                "--weightp": "", "--adapt-ltr": "",
            }
            if '-color_range' in output_dict:
                _output_dict.update({"--colorrange": output_dict["-color_range"]})
            if '-colorspace' in output_dict:
                _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            if '-color_trc' in output_dict:
                _output_dict.update({"--transfer": output_dict["-color_trc"]})
            if '-color_primaries' in output_dict:
                _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encoder:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high", "--repartition-check": "", "--trellis": "all"})
            elif "H265" in self.ARGS.render_encoder:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64", })
            if self.hdr_check_status == 2:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64",
                                     "--max-cll": "1000,100",
                                     "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"
                                     })
            _output_dict.update({"--quality": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_hwaccel_mode == "SVT":
            _input_dict = {  # '--avsw': '',
                '-fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
                '-n': f"{self.ARGS.render_gap}"
            }
            _output_dict = {
                "encc": "hevc", "-brr": "1", "-sharp": "1", "-b": ""
            }
            if "VP9" in self.ARGS.render_encoder:
                _output_dict = {
                    "encc": "vp9", "-tune": "0", "-b": ""
                }
            # TODO Color Info
            # if '-color_range' in output_dict:
            #     _output_dict.update({"--colorrange": output_dict["-color_range"]})
            # if '-colorspace' in output_dict:
            #     _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            # if '-color_trc' in output_dict:
            #     _output_dict.update({"--transfer": output_dict["-color_trc"]})
            # if '-color_primaries' in output_dict:
            #     _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'-s': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"-bit-depth": "10"})
            else:
                _output_dict.update({"-bit-depth": "8"})

            preset_mapper = {"slowest": "4", "slow": "5", "fast": "7", "faster": "9"}

            if "H265" in self.ARGS.render_encoder_preset:
                _output_dict.update({"-encMode": preset_mapper[self.ARGS.render_encoder_preset]})
            elif "VP9" in self.ARGS.render_encoder_preset:
                _output_dict.update({"-enc-mode": preset_mapper[self.ARGS.render_encoder_preset]})

            # TODO max cll, master display
            # if self.hdr_check_status == 2:
            #     _output_dict.update({"-c": "hevc",
            #                          "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
            #                          "--tier": "main", "--sao": "luma", "--ctu": "64",
            #                          "--max-cll": "1000,100",
            #                          "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"
            #                          })

            input_dict = _input_dict
            output_dict = _output_dict
            pass

        else:
            """QSV"""
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "h264_qsv",
                                    "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-rc-lookahead": "120", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_qsv",
                                    f"-g": f"{int(self.target_fps * 3)}", "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-look_ahead": "120", })

        if "ProRes" not in self.ARGS.render_encoder and self.ARGS.render_encoder_preset != "loseless":

            if self.ARGS.render_crf and self.ARGS.use_crf:
                if self.ARGS.render_hwaccel_mode != "CPU":
                    hwaccel_mode = self.ARGS.render_hwaccel_mode
                    if hwaccel_mode == "NVENC":
                        output_dict.update({"-cq:v": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "QSV":
                        output_dict.update({"-q": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "NVENCC":
                        output_dict.update({"--vbr": "0", "--vbr-quality": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "QSVENCC":
                        output_dict.update({"--la-icq": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "SVT":
                        output_dict.update({"-q": str(self.ARGS.render_crf)})

                else:  # CPU
                    output_dict.update({"-crf": str(self.ARGS.render_crf)})

            if self.ARGS.render_bitrate and self.ARGS.use_bitrate:
                if self.ARGS.render_hwaccel_mode in ["NVENCC", "QSVENCC"]:
                    output_dict.update({"--vbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                elif self.ARGS.render_hwaccel_mode == "SVT":
                    output_dict.update({"-tbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                else:
                    output_dict.update({"-b:v": f'{self.ARGS.render_bitrate}M'})
                if self.ARGS.render_hwaccel_mode == "QSV":
                    output_dict.update({"-maxrate": "200M"})

        if self.ARGS.use_manual_encode_thread:
            if self.ARGS.render_hwaccel_mode == "NVENCC":
                output_dict.update({"--output-thread": f"{self.ARGS.render_encode_thread}"})
            else:
                output_dict.update({"-threads": f"{self.ARGS.render_encode_thread}"})

        logger.debug(f"writer: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Command"""
        ffmpeg_customized_command = {}
        if len(self.ARGS.render_ffmpeg_customized):
            shlex_out = shlex.split(self.ARGS.render_ffmpeg_customized)
            if len(shlex_out) == 1:
                ffmpeg_customized_command.update({shlex_out[0]: ""})
            else:
                for i in range(len(shlex_out) - 1):
                    command = shlex_out[i]
                    next_command = shlex_out[i+1]
                    if command.startswith("-") and next_command.startswith('-'):
                        ffmpeg_customized_command.update({command: ""})
                    elif command.startswith("-"):
                        ffmpeg_customized_command.update({command: next_command})
                last_command = shlex_out[-1]
                if last_command.startswith("-"):
                    ffmpeg_customized_command.update({last_command: ""})
        logger.debug(f"ffmpeg custom: {ffmpeg_customized_command}")
        output_dict.update(ffmpeg_customized_command)
        # output_path = Tools.fillQuotation(output_path)
        if self.ARGS.render_hwaccel_mode in ["NVENCC", "QSVENCC"]:
            return EnccWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)
        elif self.ARGS.render_hwaccel_mode in ["SVT"]:
            return SVTWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)
        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    # @profile
    def check_chunk(self, del_chunk=False):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return: chunk, start_frame
        """
        if self.ARGS.is_img_output:
            """IMG OUTPUT"""
            img_io = ImgSeqIO(folder=self.project_dir, is_tool=True,
                              start_frame=-1, logger=logger,
                              output_ext=self.ARGS.output_ext, )
            last_img = img_io.get_write_start_frame()
            if self.ARGS.interp_start not in [-1, ]:
                return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)  # Manually Prioritized
            if last_img == 0:
                return 1, 0
            else:
                """last_img > 0"""
                return 1, int(last_img)

        if self.ARGS.interp_start != -1 or self.ARGS.output_chunk_cnt != -1:
            return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)

        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(self.project_dir)
        if del_chunk:
            for f in chunk_paths:
                os.remove(os.path.join(self.project_dir, f))
            return 1, 0
        if not len(chunk_paths):
            return 1, 0
        return chunk_cnt + 1, last_frame + 1

    # @profile
    def render(self, chunk_cnt, start_frame):
        """
        Render thread
        :param chunk_cnt:
        :param start_frame: render start
        :return:
        """

        def rename_chunk():
            """Maintain Chunk json"""
            if self.ARGS.is_img_output or self.main_error is not None:
                return
            chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, now_frame,
                                                                       self.output_ext)
            chunk_desc_path = os.path.join(self.project_dir, chunk_desc_path)
            if os.path.exists(chunk_desc_path):
                os.remove(chunk_desc_path)
            os.rename(chunk_tmp_path, chunk_desc_path)
            chunk_path_list.append(chunk_desc_path)
            chunk_info_path = os.path.join(self.project_dir, "chunk.json")

            with open(chunk_info_path, "w", encoding="utf-8") as w:
                chunk_info = {
                    "project_dir": self.project_dir,
                    "input": self.input,
                    "chunk_cnt": chunk_cnt,
                    "chunk_list": chunk_path_list,
                    "last_frame": now_frame,
                    "target_fps": self.target_fps,
                }
                json.dump(chunk_info, w)
            """
            key: project_dir, input filename, chunk cnt, chunk list, last frame
            """
            if is_end:
                if os.path.exists(chunk_info_path):
                    os.remove(chunk_info_path)

        # @profile
        def check_audio_concat():
            """Check Input file ext"""
            if not self.ARGS.is_save_audio or self.main_error is not None:
                return
            if self.ARGS.is_img_output:
                return
            output_ext = self.output_ext
            if "ProRes" in self.ARGS.render_encoder:
                output_ext = ".mov"

            concat_filepath = f"{os.path.join(self.output, 'concat_test')}" + output_ext
            map_audio = f'-i "{self.input}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy -shortest '
            ffmpeg_command = f'{self.ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy ' \
                             f'{Tools.fillQuotation(concat_filepath)} -y'

            logger.info("Start Audio Concat Test")
            sp = Tools.popen(ffmpeg_command)
            sp.wait()
            if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
                if self.input_ext in SupportFormat.vid_outputs:
                    logger.warning(f"Concat Test found unavailable output extension {self.output_ext}, "
                                   f"changed to {self.input_ext}")
                    self.output_ext = self.input_ext
                else:
                    logger.error(f"Concat Test Error, {output_ext}, empty output")
                    self.main_error = FileExistsError("Concat Test Error, empty output, Check Output Extension!!!")
                    raise FileExistsError(
                        "Concat Test Error, empty output detected, Please Check Your Output Extension!!!\n"
                        "e.g. mkv input should match .mkv as output extension to avoid possible concat issues")
            else:
                logger.info("Audio Concat Test Success")
                os.remove(concat_filepath)

        concat_test_flag = True

        chunk_frame_cnt = 1  # number of frames of current output chunk
        chunk_path_list = list()
        chunk_tmp_path = os.path.join(self.project_dir, f"chunk-tmp{self.output_ext}")
        frame_writer = self.generate_frame_renderer(chunk_tmp_path, start_frame)  # get frame renderer

        now_frame = start_frame
        is_end = False
        frame_written = False
        while True:
            if self.main_error is not None:
                logger.warning("Other Thread encounters Error, break")
                frame_writer.close()
                is_end = True
                rename_chunk()
                break

            frame_data = self.frames_output.get()
            if frame_data is None:
                if frame_written:
                    frame_writer.close()
                is_end = True
                rename_chunk()
                break

            now_frame = frame_data[0]
            frame = frame_data[1]

            if self.ARGS.use_fast_denoise:
                frame = cv2.fastNlMeansDenoising(frame)
            if self.ARGS.use_sr and (self.ARGS.use_sr_mode == 1 or self.ARGS.render_only):
                """先补后超"""
                frame = self.sr_module.svfi_process(frame)

            reminder_id = self.reminder_bearer.generate_reminder(30, logger, "Encoder",
                                                                 "Low Encoding speed detected, Please check your encode settings to avoid performance issues")
            if frame is not None:
                frame_written = True
                frame_writer.writeFrame(frame)
            self.reminder_bearer.terminate_reminder(reminder_id)

            chunk_frame_cnt += 1
            self.task_info.update({"chunk_cnt": chunk_cnt, "render": now_frame})  # update render info

            if not chunk_frame_cnt % self.render_gap:
                frame_writer.close()
                if concat_test_flag:
                    check_audio_concat()
                    concat_test_flag = False
                rename_chunk()
                chunk_cnt += 1
                start_frame = now_frame + 1
                frame_writer = self.generate_frame_renderer(chunk_tmp_path, start_frame)
        return

    # @profile
    def feed_to_render(self, frames_list: list, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_end: 是否是视频结尾
        :return:
        """
        frames_list_len = len(frames_list)

        for frame_i in range(frames_list_len):
            if frames_list[frame_i] is None:
                self.frames_output.put(None)
                logger.info("Put None to write_buffer in advance")
                return
            self.frames_output.put(frames_list[frame_i])  # 往输出队列（消费者）喂正常的帧
            if frame_i == frames_list_len - 1:
                if is_end:
                    self.frames_output.put(None)
                    logger.info("Put None to write_buffer")
                    return
        pass

    # @profile
    def feed_to_rife(self, now_frame: int, img0, img1, n=0, exp=0, is_end=False, add_scene=False, ):
        """
        创建任务，输出到补帧任务队列消费者
        :param now_frame:当前帧数
        :param add_scene:加入转场的前一帧（避免音画不同步和转场鬼畜）
        :param img0:
        :param img1:
        :param n:要补的帧数
        :param exp:使用指定的补帧倍率（2**exp）
        :param is_end:是否是任务结束
        :return:
        """

        def psnr(i1, i2):
            i1 = np.float64(i1)
            i2 = np.float64(i2)
            mse = np.mean((i1 - i2) ** 2)
            if mse == 0:
                return 100
            pixel_max = 255.0
            return 20 * math.log10(pixel_max / math.sqrt(mse))

        scale = self.ARGS.rife_scale
        if self.ARGS.use_rife_auto_scale:
            """使用动态光流"""
            if img0 is None or img1 is None:
                scale = 1.0
            else:
                # x = psnr(cv2.resize(img0, (256, 256)), cv2.resize(img1, (256, 256)))
                # y25 = 0.0000136703 * (x ** 3) - 0.000407396 * (x ** 2) - 0.0129 * x + 0.62621
                # y50 = 0.00000970763 * (x ** 3) - 0.0000908092 * (x ** 2) - 0.02095 * x - 0.69068
                # y100 = 0.0000134965 * (x ** 3) - 0.000246688 * (x ** 2) - 0.01987 * x - 0.70953
                # m = min(y25, y50, y100)
                # scale = {y25: 0.25, y50: 0.5, y100: 1.0}[m]
                scale = self.vfi_core.get_auto_scale(img0, img1)

        if self.ARGS.use_sr and self.ARGS.use_sr_mode == 0 and img0 is not None and img1 is not None:
            """先超后补"""
            img0, img1 = self.sr_module.svfi_process(img0), self.sr_module.svfi_process(img1)

        self.rife_task_queue.put(
            {"now_frame": now_frame, "img0": img0, "img1": img1, "n": n, "exp": exp, "scale": scale,
             "is_end": is_end, "add_scene": add_scene})

    # @profile
    def crop_read_img(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None:
            return img

        h, w, _ = img.shape
        if self.crop_param[0] > w or self.crop_param[1] > h:
            """奇怪的黑边参数，不予以处理"""
            return img
        return img[self.crop_param[1]:h - self.crop_param[1], self.crop_param[0]:w - self.crop_param[0]]

    # @profile
    def nvidia_vram_test(self):
        """
        显存测试
        :return:
        """
        try:
            if self.ARGS.resize_width and self.ARGS.resize_height:
                w, h = self.ARGS.resize_width, self.ARGS.resize_height
            else:
                w, h = list(map(lambda x: round(x), self.video_info["size"]))

            logger.info(f"Start VRAM Test: {w}x{h} with scale {self.ARGS.rife_scale}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8), \
                                   np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8)
            self.vfi_core.generate_n_interp(test_img0, test_img1, 1, self.ARGS.rife_scale)
            logger.info(f"VRAM Test Success, Resume of workflow ahead")
            del test_img0, test_img1
        except Exception as e:
            logger.error("VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc(
                limit=ArgumentManager.traceback_limit))
            raise e

    # @profile
    def remove_duplicate_frames(self, videogen_check: FFmpegReader.nextFrame, init=False) -> (list, list, dict):
        """
        获得新除重预处理帧数序列
        :param init: 第一次重复帧
        :param videogen_check:
        :return:
        """
        flow_dict = dict()
        canny_dict = dict()
        predict_dict = dict()
        resize_param = (40, 40)

        def get_img(i0):
            if i0 in check_frame_data:
                return check_frame_data[i0]
            else:
                return None

        def sobel(src):
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, -1, 3, 0, ksize=5)
            grad_y = cv2.Sobel(gray, -1, 0, 3, ksize=5)
            return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        def calc_flow_distance(pos0: int, pos1: int, _use_flow=True):
            if not _use_flow:
                return diff_canny(pos0, pos1)
            if (pos0, pos1) in flow_dict:
                return flow_dict[(pos0, pos1)]
            if (pos1, pos0) in flow_dict:
                return flow_dict[(pos1, pos0)]

            prev_gray = cv2.cvtColor(cv2.resize(get_img(pos0), resize_param), cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(cv2.resize(get_img(pos1), resize_param), cv2.COLOR_BGR2GRAY)
            flow0 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow1 = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow = (flow0 - flow1) / 2
            _x = flow[:, :, 0]
            _y = flow[:, :, 1]
            dis = np.linalg.norm(_x) + np.linalg.norm(_y)
            flow_dict[(pos0, pos1)] = dis
            return dis

        def diff_canny(pos0, pos1):
            if (pos0, pos1) in canny_dict:
                return canny_dict[(pos0, pos1)]
            if (pos1, pos0) in canny_dict:
                return canny_dict[(pos1, pos0)]
            img0, img1 = get_img(pos0), get_img(pos1)
            if self.ARGS.use_dedup_sobel:
                img0, img1 = sobel(img0), sobel(img1)
            canny_diff = cv2.Canny(cv2.absdiff(img0, img1), 100, 200).mean()
            canny_dict[(pos0, pos1)] = canny_diff
            return canny_diff

        def predict_scale(pos0, pos1):
            if (pos0, pos1) in predict_dict:
                return predict_dict[(pos0, pos1)]
            if (pos1, pos0) in predict_dict:
                return predict_dict[(pos1, pos0)]

            w, h, _ = get_img(pos0).shape
            diff = cv2.Canny(cv2.absdiff(get_img(pos0), get_img(pos0)), 100, 200)
            mask = np.where(diff != 0)
            try:
                xmin = min(list(mask)[0])
            except:
                xmin = 0
            try:
                xmax = max(list(mask)[0]) + 1
            except:
                xmax = w
            try:
                ymin = min(list(mask)[1])
            except:
                ymin = 0
            try:
                ymax = max(list(mask)[1]) + 1
            except:
                ymax = h
            W = xmax - xmin
            H = ymax - ymin
            S0 = w * h
            S1 = W * H
            prediction = -2 * (S1 / S0) + 3
            predict_dict[(pos0, pos1)] = prediction
            return prediction

        use_flow = True
        check_queue_size = max(self.frames_queue_len, 200)  # 预处理长度，非重复帧
        check_frame_list = list()  # 采样图片帧数序列,key ~ LabData
        scene_frame_list = list()  # 转场图片帧数序列,key,和check_frame_list同步
        # input_frame_data = dict()  # 输入图片数据
        check_frame_data = dict()  # 用于判断的采样图片数据
        if init:
            logger.info("Initiating Duplicated Frames Removal Process...This might take some time")
            pbar = tqdm.tqdm(total=check_queue_size, unit="frames")
        else:
            pbar = None
        """
            check_frame_list contains key, check_frame_data contains (key, frame_data)
        """
        check_frame_cnt = -1
        while len(check_frame_list) < check_queue_size:
            check_frame_cnt += 1
            check_frame = Tools.gen_next(videogen_check)
            if check_frame is None:
                break
            # check_frame_data[check_frame_cnt] = check_frame
            if len(check_frame_list):  # len>1
                # 3.5.16 Change to last type of diff
                # if self.ARGS.use_dedup_sobel:
                #     diff_result = diff_canny(check_frame_list[-1], check_frame_cnt)
                # else:
                diff_result = Tools.get_norm_img_diff(check_frame_data[check_frame_list[-1]], check_frame)
                if diff_result < 0.001:
                    # do not use pure scene check to avoid too much duplication result
                    # duplicate frames
                    continue
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Extract Frame {check_frame_cnt}")
            check_frame_data[check_frame_cnt] = check_frame
            check_frame_list.append(check_frame_cnt)  # key list
        if not len(check_frame_list):
            if init:
                pbar.close()
            return [], [], {}

        if init:
            pbar.close()
            pbar = tqdm.tqdm(total=len(check_frame_list), unit="frames")
        """Scene Batch Detection"""
        for i in range(len(check_frame_list) - 1):
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Scene Detect Frame {i}")
            i1 = check_frame_data[check_frame_list[i]]
            i2 = check_frame_data[check_frame_list[i + 1]]
            result = self.scene_detection.check_scene(i1, i2)
            if result:
                scene_frame_list.append(check_frame_list[i + 1])  # at i find scene

        if init:
            pbar.close()
            logger.info("Start Remove First Batch of Duplicated Frames")

        max_epoch = self.ARGS.remove_dup_mode  # 一直去除到一拍N，N为max_epoch，默认去除一拍二
        opt = []  # 已经被标记，识别的帧
        for queue_size, _ in enumerate(range(1, max_epoch), start=4):
            Icount = queue_size - 1  # 输入帧数
            Current = []  # 该轮被标记的帧
            i = 1
            try:
                while i < len(check_frame_list) - Icount:
                    c = [check_frame_list[p + i] for p in range(queue_size)]  # 读取queue_size帧图像 ~ 对应check_frame_list中的帧号
                    first_frame = c[0]
                    last_frame = c[-1]
                    count = 0
                    for step in range(1, queue_size - 2):
                        pos = 1
                        while pos + step <= queue_size - 2:
                            m0 = c[pos]
                            m1 = c[pos + step]
                            d0 = calc_flow_distance(first_frame, m0, use_flow)
                            d1 = calc_flow_distance(m0, m1, use_flow)
                            d2 = calc_flow_distance(m1, last_frame, use_flow)
                            value_scale = predict_scale(m0, m1)
                            if value_scale * d1 < d0 and value_scale * d1 < d2:
                                count += 1
                            pos += 1
                    if count == (queue_size * (queue_size - 5) + 6) / 2:
                        Current.append(i)  # 加入标记序号
                        i += queue_size - 3
                    i += 1
            except:
                logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            for x in Current:
                if x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
                    for t in range(queue_size - 3):
                        opt.append(t + x + 1)
        delgen = sorted(set(opt))  # 需要删除的帧
        for d in delgen:
            if check_frame_list[d] not in scene_frame_list:
                check_frame_list[d] = -1

        max_key = np.max(list(check_frame_data.keys()))
        if max_key not in check_frame_list:
            check_frame_list.append(max_key)
        if 0 not in check_frame_list:
            check_frame_list.insert(0, 0)
        check_frame_list = [i for i in check_frame_list if i > -1]
        return check_frame_list, scene_frame_list, check_frame_data

    def rife_run_rest(self, run_time: float):
        rest_exp = 3600
        if self.ARGS.multi_task_rest and self.ARGS.multi_task_rest_interval and \
                time.time() - run_time > self.ARGS.multi_task_rest_interval * rest_exp:
            logger.info(
                f"\n\n INFO - Exceed Run Interval {self.ARGS.multi_task_rest_interval} hour. Time to Rest for 5 minutes!")
            time.sleep(600)
            return time.time()
        return run_time

    def rife_run_input_check(self, dedup=False):
        """
        perform input availability check and return generator of frames
        :return: chunk_cnt, start_frame, videogen, videogen_check
        """
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        logger.info("Resuming Video Frames...")

        """Get Frames to interpolate"""
        reminder_id = self.reminder_bearer.generate_reminder(300, logger, "Decode Input",
                                                             "Please consider terminate current process manually, check input arguments and restart. It's normal to wait for at least 30 minutes for 4K input when performing resume of workflow")
        videogen = self.generate_frame_reader(start_frame).nextFrame()
        videogen_check = None
        if dedup:
            videogen_check = self.generate_frame_reader(start_frame, frame_check=True).nextFrame()
        videogen_available_check = self.generate_frame_reader(start_frame, frame_check=True).nextFrame()

        check_img1 = self.crop_read_img(Tools.gen_next(videogen_available_check))
        self.reminder_bearer.terminate_reminder(reminder_id)
        videogen_available_check.close()
        if check_img1 is None:
            self.main_error = OSError(
                f"Input file is not available: {self.input}, is img input: {self.ARGS.is_img_input},"
                f"Please Check Your Input Settings"
                f"(Start Chunk, Start Frame, Start Point, Start Frame)")
            self.rife_work_event.set()
            raise self.main_error
        return chunk_cnt, start_frame, videogen, videogen_check

    # @profile
    def rife_run(self):
        """
        Go through all procedures to produce interpolation result in dedup mode
        :return:
        """

        logger.info("Activate Remove Duplicate Frames Mode")
        chunk_cnt, now_frame_key, videogen, videogen_check = self.rife_run_input_check(dedup=True)
        logger.info("Loaded Input Frames")
        is_end = False

        self.rife_work_event.set()
        """Start Process"""
        run_time = time.time()
        first_run = True
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                logger.critical("Render Thread Dead Unexpectedly")
                break

            run_time = self.rife_run_rest(run_time)

            check_frame_list, scene_frame_list, input_frame_data = self.remove_duplicate_frames(videogen_check,
                                                                                                init=first_run)
            input_frame_data = dict(input_frame_data)
            first_run = False
            if not len(check_frame_list):
                while True:
                    img1 = self.crop_read_img(Tools.gen_next(videogen))
                    if img1 is None:
                        is_end = True
                        self.feed_to_rife(now_frame_key, img1, img1, n=0,
                                          is_end=is_end)
                        break
                    self.feed_to_rife(now_frame_key, img1, img1, n=0)
                break

            else:
                img0 = self.crop_read_img(Tools.gen_next(videogen))
                img1 = img0.copy()
                last_frame_key = check_frame_list[0]
                now_a_key = last_frame_key
                for frame_cnt in range(1, len(check_frame_list)):
                    now_b_key = check_frame_list[frame_cnt]
                    img1 = img0.copy()
                    """A - Interpolate -> B"""
                    while True:
                        last_possible_scene = img1
                        if now_a_key != now_b_key:
                            img1 = self.crop_read_img(Tools.gen_next(videogen))
                            now_a_key += 1
                        else:
                            break
                    now_frame_key = now_b_key
                    self.task_info.update({"now_frame": now_frame_key})
                    if now_frame_key in scene_frame_list:
                        self.scene_detection.update_scene_status(now_frame_key, "scene")
                        potential_key = now_frame_key - 1
                        if potential_key > 0 and potential_key in input_frame_data:
                            before_img = last_possible_scene
                        else:
                            before_img = img0

                        # Scene Review, should be annoted
                        # title = f"try:"
                        # comp_stack = np.hstack((img0, before_img, img1))
                        # comp_stack = cv2.resize(comp_stack, (1440, 270))
                        # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
                        # cv2.moveWindow(title, 0, 0)
                        # cv2.resizeWindow(title, 1440, 270)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        if frame_cnt < 1:
                            self.feed_to_rife(now_frame_key, img0, img0, n=0,
                                              is_end=is_end)
                        elif self.ARGS.is_scdet_mix:
                            self.feed_to_rife(now_frame_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                              add_scene=True,
                                              is_end=is_end)
                        else:
                            self.feed_to_rife(now_frame_key, img0, before_img, n=now_frame_key - last_frame_key - 2,
                                              add_scene=True,
                                              is_end=is_end)
                    else:
                        self.scene_detection.update_scene_status(now_frame_key, "normal")
                        self.feed_to_rife(now_b_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                          is_end=is_end)
                    last_frame_key = now_frame_key
                    img0 = img1
                self.feed_to_rife(now_frame_key, img1, img1, n=0, is_end=is_end)
                self.task_info.update({"now_frame": check_frame_list[-1]})

        pass
        self.rife_task_queue.put(None)
        videogen.close()
        videogen_check.close()
        """Wait for Rife and Render Thread to finish"""

    # @profile
    def rife_run_any_fps(self):
        """
        Go through all procedures to produce interpolation result in any fps mode(from a fps to b fps)
        :return:
        """

        logger.info("Activate Any FPS Mode")
        chunk_cnt, now_frame, videogen, videogen_check = self.rife_run_input_check(dedup=True)
        img1 = self.crop_read_img(Tools.gen_next(videogen))
        logger.info("Loaded Input Frames")
        is_end = False

        """Update Interp Mode Info"""
        if self.ARGS.remove_dup_mode == 1:  # 单一模式
            self.ARGS.remove_dup_threshold = self.ARGS.remove_dup_threshold if self.ARGS.remove_dup_threshold > 0.01 else 0.01
        else:  # 0， 不去除重复帧
            self.ARGS.remove_dup_threshold = 0.001

        self.rife_work_event.set()
        """Start Process"""
        run_time = time.time()
        while True:
            if is_end or self.main_error:
                break

            if not self.render_thread.is_alive():
                logger.critical("Render Thread Dead Unexpectedly")
                break

            run_time = self.rife_run_rest(run_time)

            img0 = img1
            img1 = self.crop_read_img(Tools.gen_next(videogen))

            now_frame += 1

            if img1 is None:
                is_end = True
                self.feed_to_rife(now_frame, img0, img0, is_end=is_end)
                break

            diff = Tools.get_norm_img_diff(img0, img1)
            skip = 0  # 用于记录跳过的帧数

            """Find Scene"""
            if self.scene_detection.check_scene(img0, img1):
                self.feed_to_rife(now_frame, img0, img1, n=0,
                                  is_end=is_end)  # add img0 only, for there's no gap between img0 and img1
                self.scene_detection.update_scene_status(now_frame, "scene")
                continue
            else:
                if diff < self.ARGS.remove_dup_threshold:
                    before_img = img1.copy()
                    is_scene = False
                    while diff < self.ARGS.remove_dup_threshold:
                        skip += 1
                        self.scene_detection.update_scene_status(now_frame, "dup")
                        last_frame = img1.copy()
                        img1 = self.crop_read_img(Tools.gen_next(videogen))

                        if img1 is None:
                            img1 = last_frame
                            is_end = True
                            break

                        diff = Tools.get_norm_img_diff(img0, img1)

                        is_scene = self.scene_detection.check_scene(img0, img1)  # update scene stack
                        if is_scene:
                            break
                        if skip == self.dup_skip_limit * self.target_fps // self.input_fps:
                            """超过重复帧计数限额，直接跳出"""
                            break

                    # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                    if is_scene:
                        if self.ARGS.is_scdet_mix:
                            self.feed_to_rife(now_frame, img0, img1, n=skip, add_scene=True,
                                              is_end=is_end)
                        else:
                            self.feed_to_rife(now_frame, img0, before_img, n=skip - 1, add_scene=True,
                                              is_end=is_end)
                            """
                            0 (1 2 3) 4[scene] => 0 (1 2) 3 4[scene] 括号内为RIFE应该生成的帧
                            """
                        self.scene_detection.update_scene_status(now_frame, "scene")

                    elif skip != 0:  # skip >= 1
                        assert skip >= 1
                        """Not Scene"""
                        self.feed_to_rife(now_frame, img0, img1, n=skip, is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "normal")
                    now_frame += skip + 1
                else:
                    """normal frames"""
                    self.feed_to_rife(now_frame, img0, img1, n=0, is_end=is_end)  # 当前模式下非重复帧间没有空隙，仅输入img0
                    self.scene_detection.update_scene_status(now_frame, "normal")
                self.task_info.update({"now_frame": now_frame})
            pass

        self.rife_task_queue.put(None)  # bad way to end

    # @profile
    def run(self):
        run_all_time = datetime.datetime.now()

        """Check Steam Validation"""
        if self.ARGS.is_steam:
            if not self.STEAM.steam_valid:
                error = str(self.STEAM.steam_error).split('\n')[-1]
                logger.error(f"Steam Validation Failed: {error}")
                return
            else:
                valid_response = self.STEAM.CheckSteamAuth()
                if valid_response != 0:
                    logger.error(f"Steam Validation Failed, code {valid_response}")
                    return
            steam_dlc_check = self.STEAM.CheckProDLC(0)
            if not steam_dlc_check:
                _msg = "SVFI - Pro DLC Not Purchased,"
                if self.ARGS.extract_only or self.ARGS.render_only:
                    raise GenericSteamException(f"{_msg} Extract/Render ToolBox Unavailable")
                if self.ARGS.input_start_point is not None or self.ARGS.input_end_point is not None:
                    raise GenericSteamException(f"{_msg} Manual Input Section Unavailable")
                if self.ARGS.is_scdet_output or self.ARGS.is_scdet_mix:
                    raise GenericSteamException(f"{_msg} Scdet Output/Mix Unavailable")
                if self.ARGS.use_sr:
                    raise GenericSteamException(f"{_msg} Super Resolution Module Unavailable")
                if self.ARGS.use_rife_multi_cards:
                    raise GenericSteamException(f"{_msg} Multi Video Cards Work flow Unavailable")

        """Go through the process"""
        if self.ARGS.concat_only:
            # self.project_dir = self.input
            # self.ARGS.is_img_input = False
            self.concat_all()
        elif self.ARGS.extract_only:
            self.extract_only()
            pass
        elif self.ARGS.render_only:
            self.render_only()
            pass
        else:
            def update_progress():
                nonlocal previous_cnt
                scene_status = self.scene_detection.get_scene_status()

                render_status = self.task_info  # render status quo
                """(chunk_cnt, start_frame, end_frame, frame_cnt)"""

                pbar.set_description(
                    f"Process at Chunk {render_status['chunk_cnt']:0>3d}")
                pbar.set_postfix({"R": f"{render_status['render']}", "C": f"{now_frame}",
                                  "S": f"{scene_status['recent_scene']}",
                                  "SC": f"{self.scene_detection.scdet_cnt}", "TAT": f"{task_acquire_time:.2f}s",
                                  "PT": f"{process_time:.2f}s", "QL": f"{self.rife_task_queue.qsize()}"})
                pbar.update(now_frame - previous_cnt)
                previous_cnt = now_frame
                pass

            """Concat Already / Mission Conflict Check & Dolby Vision Sort"""
            concat_filepath, output_ext = self.get_output_path()
            if os.path.exists(concat_filepath):
                logger.warning("Mission Already Finished, "
                               "Jump to Dolby Vision Check")
                if self.hdr_check_status == 3:
                    """Dolby Vision"""
                    dovi_maker = DoviProcesser(concat_filepath, logger, self.project_dir, self.ARGS,
                                               self.interp_exp)
                    dovi_maker.run()
            else:
                """Load RIFE Model"""
                if self.ARGS.use_ncnn:
                    self.ARGS.rife_model_name = os.path.basename(self.ARGS.rife_model)
                    from Utils import inference_rife_ncnn as inference
                else:
                    try:
                        # raise Exception("Load Torch Failed Test")
                        from Utils import inference_rife as inference
                    except Exception:
                        logger.warning("Import Torch Failed, use NCNN-RIFE instead")
                        logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
                        self.ARGS.use_ncnn = True
                        self.ARGS.rife_model = "rife-v2"
                        self.ARGS.rife_model_name = "rife-v2"
                        from Utils import inference_rife_ncnn as inference

                """Update RIFE Core"""
                self.vfi_core = inference.RifeInterpolation(self.ARGS)
                self.vfi_core.initiate_algorithm(self.ARGS)

                if not self.ARGS.use_ncnn:
                    self.nvidia_vram_test()

                """Get RIFE Task Thread"""
                if self.ARGS.remove_dup_mode in [0, 1]:
                    self.rife_thread = threading.Thread(target=self.rife_run_any_fps, name="[ARGS] RifeTaskThread", )
                else:  # 1, 2 => 去重一拍二或一拍三
                    self.rife_thread = threading.Thread(target=self.rife_run, name="[ARGS] RifeTaskThread", )
                self.rife_thread.start()

                """Get Renderer"""
                chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
                self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                                      args=(chunk_cnt, start_frame,))
                self.render_thread.start()

                previous_cnt = start_frame
                now_frame = start_frame
                PURE_SCENE_THRESHOLD = 30

                self.rife_work_event.wait()  # 等待补帧线程启动（等待all frames cnt 更新、验证啥的）
                if self.main_error:
                    logger.error("Threads outside RUN encounters error,")
                    self.feed_to_render([None], is_end=True)
                    raise self.main_error

                pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
                pbar.update(n=start_frame)
                pbar.unpause()
                task_acquire_time = time.time()
                process_time = time.time()
                while True:
                    # if not self.rife_thread.is_alive():
                    #     raise AssertionError("RIFE Thread Dead Unexpectedly without putting none to buffer")
                    task = self.rife_task_queue.get()
                    task_acquire_time = time.time() - task_acquire_time
                    if task is None:
                        self.feed_to_render([None], is_end=True)
                        break
                    """
                    task = {"now_frame", "img0", "img1", "n", "exp","scale", "is_end", "is_scene", "add_scene"}
                    """
                    # now_frame = task["now_frame"]
                    img0 = task["img0"]
                    img1 = task["img1"]
                    n = task["n"]
                    scale = task["scale"]
                    is_end = task["is_end"]
                    add_scene = task["add_scene"]

                    debug = False
                    """Test
                    1. 正常4K，解码编码
                    2. 一拍N卡顿
                    """

                    if img1 is None:
                        self.feed_to_render([None], is_end=True)
                        break

                    frames_list = [img0]
                    if self.ARGS.is_scdet_mix and add_scene:
                        mix_list = Tools.get_mixed_scenes(img0, img1, n + 1)
                        frames_list.extend(mix_list)
                    else:
                        reminder_id = self.reminder_bearer.generate_reminder(60, logger,
                                                                             "Video Frame Interpolation",
                                                                             "Low interpolate speed detected, Please consider lower your output settings to enhance speed")
                        if n > 0:
                            if n > PURE_SCENE_THRESHOLD and Tools.check_pure_img(img0):
                                """It's Pure Img Sequence, Copy img0"""
                                for i in range(n):
                                    frames_list.append(img0)
                            else:
                                interp_list = self.vfi_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                                frames_list.extend(interp_list)
                        if add_scene:  # [AA BBB CC DDD] E
                            frames_list.append(img1)
                        self.reminder_bearer.terminate_reminder(reminder_id)
                    feed_list = list()
                    for i in frames_list:
                        feed_list.append([now_frame, i])
                        now_frame += 1
                    if self.ARGS.use_evict_flicker or self.ARGS.use_rife_fp16:
                        img_ori = frames_list[0].copy()
                        frames_list[0] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                         debug=debug)
                        if add_scene:
                            img_ori = frames_list[-1].copy()
                            frames_list[-1] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                              debug=debug)

                    self.feed_to_render(feed_list, is_end=is_end)
                    process_time = time.time() - process_time
                    update_progress()
                    process_time = time.time()
                    task_acquire_time = time.time()
                    if is_end:
                        break

                process_time = 0  # rife's work is done
                task_acquire_time = 0  # task acquire is impossible
                while (self.render_thread is not None and self.render_thread.is_alive()) or \
                        (self.rife_thread is not None and self.rife_thread.is_alive()):
                    """等待渲染线程结束"""
                    update_progress()
                    time.sleep(0.1)

                pbar.update(abs(self.all_frames_cnt - now_frame))
                pbar.close()

                logger.info(f"Scedet Status Quo: {self.scene_detection.get_scene_status()}")

                """Check Finished Safely"""
                if self.main_error is not None:
                    raise self.main_error

                """Concat the chunks"""
                if not self.ARGS.is_no_concat and not self.ARGS.is_img_output:
                    self.concat_all()

                self.steam_update_achv()  # TODO Check Steam ACHV available for render only etc.

        # if os.path.exists(self.ARGS.config):
        #     logger.info("Successfully Remove Config File")
        #     os.remove(self.ARGS.config)
        logger.info(f"Program finished at {datetime.datetime.now()}: "
                    f"Duration: {datetime.datetime.now() - run_all_time}")
        logger.info("Please Note That Merchandise Use of SVFI's Output is Strictly PROHIBITED, "
                    "Check EULA for more details")
        self.reminder_bearer.terminate_all()
        utils_overtime_reminder_bearer.terminate_all()
        pass

    def steam_update_achv(self):
        """
        Update Steam Achievement
        :return:
        """
        if not self.ARGS.is_steam or self.main_error is not None:
            """If encountered serious error in the process, end steam update"""
            return
        """Get Stat"""
        STAT_INT_FINISHED_CNT = self.STEAM.GetStat("STAT_INT_FINISHED_CNT", int)
        STAT_FLOAT_FINISHED_MINUTE = self.STEAM.GetStat("STAT_FLOAT_FINISHED_MIN", float)

        """Update Stat"""
        STAT_INT_FINISHED_CNT += 1
        reply = self.STEAM.SetStat("STAT_INT_FINISHED_CNT", STAT_INT_FINISHED_CNT)
        if self.all_frames_cnt >= 0 and not self.ARGS.render_only:
            """Update Mission Process Time only in interpolation"""
            STAT_FLOAT_FINISHED_MINUTE += self.all_frames_cnt / self.target_fps / 60
            reply = self.STEAM.SetStat("STAT_FLOAT_FINISHED_MIN", round(STAT_FLOAT_FINISHED_MINUTE, 2))

        """Get ACHV"""
        ACHV_Task_Frozen = self.STEAM.GetAchv("ACHV_Task_Frozen")
        ACHV_Task_Cruella = self.STEAM.GetAchv("ACHV_Task_Cruella")
        ACHV_Task_Suzumiya = self.STEAM.GetAchv("ACHV_Task_Suzumiya")
        ACHV_Task_1000M = self.STEAM.GetAchv("ACHV_Task_1000M")
        ACHV_Task_10 = self.STEAM.GetAchv("ACHV_Task_10")
        ACHV_Task_50 = self.STEAM.GetAchv("ACHV_Task_50")

        """Update ACHV"""
        output_path, _ = self.get_output_path()
        if 'Frozen' in output_path and not ACHV_Task_Frozen:
            reply = self.STEAM.SetAchv("ACHV_Task_Frozen")
        if 'Cruella' in output_path and not ACHV_Task_Cruella:
            reply = self.STEAM.SetAchv("ACHV_Task_Cruella")
        if any([i in output_path for i in ['Suzumiya', 'Haruhi', '涼宮', '涼宮ハルヒの憂鬱', '涼宮ハルヒの消失', '凉宫春日']]) \
                and not ACHV_Task_Suzumiya:
            reply = self.STEAM.SetAchv("ACHV_Task_Suzumiya")
        if STAT_INT_FINISHED_CNT > 10 and not ACHV_Task_10:
            reply = self.STEAM.SetAchv("ACHV_Task_10")
        if STAT_INT_FINISHED_CNT > 50 and not ACHV_Task_50:
            reply = self.STEAM.SetAchv("ACHV_Task_50")
        if STAT_FLOAT_FINISHED_MINUTE > 1000 and not ACHV_Task_1000M:
            reply = self.STEAM.SetAchv("ACHV_Task_1000M")
        self.STEAM.Store()
        pass

    def extract_only(self):
        if self.output_ext not in SupportFormat.img_outputs:
            self.output_ext = ".png"
            logger.warning("Auto change output extension to png")
        chunk_cnt, start_frame = self.check_chunk()
        videogen = self.generate_frame_reader(start_frame).nextFrame()

        img1 = self.crop_read_img(Tools.gen_next(videogen))
        if img1 is None:
            self.reminder_bearer.terminate_all()
            raise OSError(f"Input file is not available: {self.input}, is img input: {self.ARGS.is_img_input},"
                          f"Please Check Your Input Settings"
                          f"(Start Chunk, Start Frame, Start Point, Start Frame)")

        renderer = ImgSeqIO(folder=self.project_dir, is_read=False,
                            start_frame=-1, logger=logger,  # auto write to destination
                            output_ext=self.ARGS.output_ext, )
        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        img_cnt = 0
        while img1 is not None:
            renderer.writeFrame(img1)
            pbar.update(n=1)
            img_cnt += 1
            pbar.set_description(
                f"Process at Extracting Img {img_cnt}")
            img1 = self.crop_read_img(Tools.gen_next(videogen))

        renderer.close()

    def render_only(self):
        def update_progress():
            nonlocal render_cnt
            render_status = self.task_info  # render status quo
            pbar.set_description(f"Process at Rendering Chunk {render_status['chunk_cnt']}")
            pbar.update(render_status['render'] - render_cnt)
            render_cnt = render_status['render']
            pass

        chunk_cnt, start_frame = self.check_chunk()
        videogen = self.generate_frame_reader(start_frame).nextFrame()
        self.render_thread = threading.Thread(target=self.render, name="[ARGS] RenderThread",
                                              args=(chunk_cnt, start_frame,))
        self.render_thread.start()

        img1 = self.crop_read_img(Tools.gen_next(videogen))
        if img1 is None:
            raise OSError(f"Input file is not available: {self.input}, is img input: {self.ARGS.is_img_input},"
                          f"Please Check Your Input Settings"
                          f"(Start Chunk, Start Frame, Start Point, Start Frame)")

        pbar = tqdm.tqdm(total=self.all_frames_cnt, unit="frames")
        pbar.update(n=start_frame)
        previous_cnt = start_frame
        render_cnt = start_frame

        while img1 is not None:
            previous_cnt += 1
            update_progress()
            self.feed_to_render([[previous_cnt, img1]])
            img1 = self.crop_read_img(Tools.gen_next(videogen))
        self.feed_to_render([None], is_end=True)

        while self.render_thread is not None and self.render_thread.is_alive():
            """等待渲染线程结束"""
            update_progress()
            time.sleep(0.1)

        if self.main_error is not None:
            return

        """Concat the chunks"""
        if not self.ARGS.is_no_concat and not self.ARGS.is_img_output:
            self.concat_all()

    def get_output_path(self):
        """
        Get Output Path for Process
        :return:
        """
        """Check Input file ext"""
        output_ext = self.output_ext
        if "ProRes" in self.ARGS.render_encoder:
            output_ext = ".mov"

        output_filepath = f"{os.path.join(self.output, Tools.get_filename(self.input))}"
        if self.ARGS.render_only:
            output_filepath += "_SVFI_Render"  # 仅渲染
        output_filepath += f"_{int(self.target_fps)}fps"  # 补帧

        if self.ARGS.is_render_slow_motion:  # 慢动作
            output_filepath += f"_[SLM_{self.ARGS.render_slow_motion_fps}fps]"
        if self.ARGS.use_deinterlace:
            output_filepath += f"_[DI]"
        if self.ARGS.use_fast_denoise:
            output_filepath += f"_[DN]"

        if not self.ARGS.render_only:
            """RIFE"""
            if self.ARGS.use_rife_auto_scale:
                output_filepath += f"_[SA]"
            else:
                output_filepath += f"_[S-{self.ARGS.rife_scale}]"  # 全局光流尺度
            if self.ARGS.use_ncnn:
                output_filepath += "_[NCNN]"
            output_filepath += f"_[{os.path.basename(self.ARGS.rife_model_name)}]"  # 添加模型信息
            if self.ARGS.use_rife_fp16:
                output_filepath += "_[FP16]"
            if self.ARGS.is_rife_reverse:
                output_filepath += "_[RR]"
            if self.ARGS.use_rife_forward_ensemble:
                output_filepath += "_[RFE]"
            if self.ARGS.rife_tta_mode:
                output_filepath += f"_[TTA-{self.ARGS.rife_tta_mode}-{self.ARGS.rife_tta_iter}]"
            if self.ARGS.remove_dup_mode:  # 去重模式
                output_filepath += f"_[RD-{self.ARGS.remove_dup_mode}]"

        if self.ARGS.use_sr:  # 使用超分
            sr_model = os.path.splitext(self.ARGS.use_sr_model)[0]
            output_filepath += f"_[SR-{self.ARGS.use_sr_algo}-{sr_model}]"

        output_filepath += f"_{self.ARGS.task_id[-6:]}"
        output_filepath += output_ext  # 添加后缀名
        return output_filepath, output_ext

    # @profile
    @overtime_reminder_deco(300, logger, "Concat Chunks",
                            "This is normal for long footage more than 30 chunks, please wait patiently until concat is done")
    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.project_dir)
        concat_path = os.path.join(self.project_dir, "concat.ini")
        logger.info("Final Round Finished, Start Concating")
        concat_list = list()

        for f in os.listdir(self.project_dir):
            if re.match("chunk-\d+-\d+-\d+", f):
                concat_list.append(os.path.join(self.project_dir, f))
            else:
                logger.debug(f"concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if not len(concat_path):
            raise OSError(
                f"Could not find any chunks, the chunks could have already been concatenated or removed, please check your output folder.")

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        concat_filepath, output_ext = self.get_output_path()

        if self.ARGS.is_save_audio and not self.ARGS.is_img_input:
            audio_path = self.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -map 1:s? -c:a copy -c:s copy '
            if self.ARGS.input_start_point or self.ARGS.input_end_point:
                map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a? -c:a aac -ab 640k '
                if self.ARGS.input_end_point is not None:
                    map_audio = f'-to {self.ARGS.input_end_point} {map_audio}'
                if self.ARGS.input_start_point is not None:
                    map_audio = f'-ss {self.ARGS.input_start_point} {map_audio}'

            if self.input_ext in ['.vob'] and self.output_ext in ['.mkv']:
                map_audio += "-map_chapters -1 "

        else:
            map_audio = ""

        color_info_str = ' '.join(Tools.dict2Args(self.color_info))

        ffmpeg_command = f'{self.ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy ' \
                         f'{Tools.fillQuotation(concat_filepath)} -metadata title="Powered By SVFI {self.ARGS.version}" ' \
                         f'{color_info_str} ' \
                         f'-y'

        logger.debug(f"Concat command: {ffmpeg_command}")
        sp = Tools.popen(ffmpeg_command)
        sp.wait()
        logger.info(f"Concat {len(concat_list)} files to {os.path.basename(concat_filepath)}")
        if self.hdr_check_status == 3:
            logger.info("Start DOVI Conversion")
            dovi_maker = DoviProcesser(concat_filepath, logger, self.project_dir, self.ARGS,
                                       self.interp_exp)
            dovi_maker.run()
        if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
            logger.error(f"Concat Error, with output extension {output_ext}")
            raise FileExistsError(
                f"Concat Error with output extension {output_ext}, empty output detected, Please Check Your Output Extension!!!\n"
                "e.g. mkv input should match .mkv as output extension to avoid possible concat issues")
        if self.ARGS.is_output_only:
            self.check_chunk(del_chunk=True)

    def concat_check(self, concat_list, concat_filepath):
        """
        Check if concat output is valid
        :param concat_filepath:
        :param concat_list:
        :return:
        """
        original_concat_size = 0
        for f in concat_list:
            original_concat_size += os.path.getsize(f)
        output_concat_size = os.path.getsize(concat_filepath)
        if output_concat_size < original_concat_size * 0.9:
            return False
        return True


interpworkflow = InterpWorkFlow(ARGS)
interpworkflow.run()
sys.exit(0)
