# -*- coding: utf-8 -*-
import datetime
import glob
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess as sp
import sys
import time
import traceback

import cv2
import psutil
import torch
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QSettings, pyqtSignal, pyqtSlot, QThread, QTime, QVariant, QPoint, QSize
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication, QMessageBox, QFileDialog

from Utils import SVFI_UI, SVFI_help, SVFI_about, SVFI_preference, SVFI_preview_args
from Utils.RIFE_GUI_Custom import SVFI_Config_Manager, SVFITranslator
from Utils.utils import Tools, EncodePresetAssemply, ImgSeqIO, SupportFormat, ArgumentManager, SteamUtils, appDir

MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False

Utils = Tools()
appDataPath = os.path.join(appDir, "SVFI.ini")
appPrefPath = os.path.join(appDir, "SVFI_Preference.ini")
appData = QSettings(appDataPath, QSettings.IniFormat)
appData.setIniCodec("UTF-8")
appPref = QSettings(appPrefPath, QSettings.IniFormat)
appPref.setIniCodec("UTF-8")

logger = Utils.get_logger("GUI", appDir)
ols_potential = os.path.join(appDir, "one_line_shot_args.exe")
ico_path = os.path.join(appDir, "svfi.ico")

translator = SVFITranslator()


def _translate(from_where='@default', input_text=""):
    return QCoreApplication.translate('', input_text)


class UiHelpDialog(QDialog, SVFI_help.Ui_Dialog):
    def __init__(self, parent=None):
        super(UiHelpDialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)


class UiAboutDialog(QDialog, SVFI_about.Ui_Dialog):
    def __init__(self, parent=None):
        super(UiAboutDialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)


class UiPreviewArgsDialog(QDialog, SVFI_preview_args.Ui_Dialog):
    def __init__(self, parent=None):
        super(UiPreviewArgsDialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)
        self.default_args = self.ArgsLabel.text()
        self.args_list = re.findall("{(.*?)}", self.default_args)
        self.ArgumentsPreview()

    def ArgumentsPreview(self):
        """
        Generate String for Label of Arguments' Preview
        :return:
        """
        args_string = str(self.default_args)
        for arg_key in self.args_list:
            arg_data = appData.value(arg_key, "")
            _arg = ""
            if isinstance(arg_data, list):
                _arg = "\n".join(arg_data)
            else:
                _arg = str(arg_data)
            args_string = args_string.replace(f"{{{arg_key}}}", html.escape(_arg))

        logger.debug(f"Check Arguments Preview: \n{args_string}")
        self.ArgsLabel.setText(args_string)


class UiPreferenceDialog(QDialog, SVFI_preference.Ui_Dialog):
    preference_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(UiPreferenceDialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)
        self.update_preference()
        self.ExpertModeChecker.clicked.connect(self.request_preference)
        self.buttonBox.clicked.connect(self.request_preference)

    def closeEvent(self, event):
        self.request_preference()

    def close(self):
        self.request_preference()

    def update_preference(self):
        """
        初始化，更新偏好设置
        :return:
        """
        self.MultiTaskRestChecker.setChecked(appPref.value("multi_task_rest", False, type=bool))
        self.MultiTaskRestInterval.setValue(appPref.value("multi_task_rest_interval", 0, type=int))
        self.AfterMission.setCurrentIndex(appPref.value("after_mission", False, type=bool))  # None
        self.ForceCpuChecker.setChecked(appPref.value("rife_use_cpu", False, type=bool))
        self.ExpertModeChecker.setChecked(appPref.value("expert", False, type=bool))
        self.PreviewArgsModeChecker.setChecked(appPref.value("is_preview_args", False, type=bool))
        self.RudeExitModeChecker.setChecked(appPref.value("is_rude_exit", True, type=bool))
        self.QuietModeChecker.setChecked(appPref.value("is_gui_quiet", False, type=bool))
        self.WinOnTopChecker.setChecked(appPref.value("is_windows_ontop", False, type=bool))
        self.OneWayModeChecker.setChecked(appPref.value("use_clear_inputs", False, type=bool))
        self.UseGlobalSettingsChecker.setChecked(appPref.value("use_global_settings", False, type=bool))
        pass

    def request_preference(self):
        """
        申请偏好设置更改
        :return:
        """
        appPref.setValue("multi_task_rest", self.MultiTaskRestChecker.isChecked())
        appPref.setValue("multi_task_rest_interval", self.MultiTaskRestInterval.value())
        appPref.setValue("after_mission", self.AfterMission.currentIndex())
        appPref.setValue("rife_use_cpu", self.ForceCpuChecker.isChecked())
        appPref.setValue("expert", self.ExpertModeChecker.isChecked())
        appPref.setValue("is_preview_args", self.PreviewArgsModeChecker.isChecked())
        appPref.setValue("is_rude_exit", self.RudeExitModeChecker.isChecked())
        appPref.setValue("is_gui_quiet", self.QuietModeChecker.isChecked())
        appPref.setValue("is_windows_ontop", self.WinOnTopChecker.isChecked())
        appPref.setValue("use_clear_inputs", self.OneWayModeChecker.isChecked())
        appPref.setValue("use_global_settings", self.UseGlobalSettingsChecker.isChecked())
        self.preference_signal.emit({})


class UiRunThread(QThread):
    run_signal = pyqtSignal(dict)

    def __init__(self, command, task_id=0, data=None, parent=None):
        """
        多线程运行系统命令
        :param command:
        :param task_id:
        :param data: 信息回传时的数据
        :param parent:
        """
        super(UiRunThread, self).__init__(parent)
        self.command = command
        self.task_id = task_id
        self.data = data

    def fire_finish_signal(self):
        emit_json = {"id": self.task_id, "status": 1, "data": self.data}
        self.run_signal.emit(emit_json)

    def run(self):
        logger.info(f"[CMD Thread]: Start execute {self.command}")
        ps = Tools.popen(self.command)
        ps.wait()
        self.fire_finish_signal()
        pass


class UiRun(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, parent=None, concat_only=False, extract_only=False, render_only=False, task_list: list = None):
        """
        Launch Task Thread
        :param parent:
        :param concat_only:
        :param extract_only:
        :param render_only:
        :param task_list: [int], only execute selected task
        """
        super(UiRun, self).__init__(parent)
        self.concat_only = concat_only
        self.extract_only = extract_only
        self.render_only = render_only
        self.task_list = task_list
        self.command = ""
        self.current_proc = None
        self.kill = False
        self.pause = False
        self.task_cnt = 0
        self.silent = False
        self.current_filename = ""
        self.current_step = 0
        self.main_error = ""

    def build_command(self, item_data: dict) -> (str, str):
        global appData
        config_manager = SVFI_Config_Manager(item_data, appDir)
        config_path = config_manager.FetchConfig()
        if config_path is None:
            logger.error(f"Invalid Task: {item_data}")
            return None, ""

        if appData is None:
            logger.error(f"Invalid appData, check previous mission load")

        ols_paths = os.path.splitext(ols_potential)
        if ols_paths[-1] == ".exe":
            self.command = ols_potential + " "
        else:
            """python script"""
            self.command = f'python "{ols_potential}" '

        input_path = item_data['input_path']
        task_id = item_data['task_id']

        self.command += f'--input {Tools.fillQuotation(input_path)} --task-id {task_id} '
        self.command += f'--config {Tools.fillQuotation(config_path)} '

        """Alternative Mission Settings"""
        if self.concat_only:
            self.command += f"--concat-only "
        if self.extract_only:
            self.command += f"--extract-only "
        if self.render_only:
            self.command += f"--render-only "

        self.command = self.command.replace("\\", "/")
        return input_path, self.command

    def update_status(self, finished=False, notice="", sp_status="", returncode=-1):
        """
        update sub process status
        :return:
        """
        emit_json = {"cnt": len(self.task_list), "current": self.current_step, "finished": finished,
                     "notice": notice, "subprocess": sp_status, "returncode": returncode}
        emit_json = json.dumps(emit_json)
        self.run_signal.emit(emit_json)

    @staticmethod
    def maintain_multitask():
        pass

    def run(self):
        try:
            logger.info("SVFI Task Run")
            try:
                input_list_data = json.loads(appPref.value("gui_inputs", "{}"))
            except json.decoder.JSONDecodeError:
                logger.info("Failed to execute RIFE Tasks as there are no valid gui_inputs, check appData")
                self.update_status(True, returncode=502)
                return

            command_list = list()
            for item_data in input_list_data['inputs']:
                input_path, command = self.build_command(item_data)
                command_list.append((input_path, command))

            self.current_step = 0
            self.task_cnt = len(command_list)
            if self.task_list is None or not len(self.task_list):
                logger.info("Add All tasks into queue")
                self.task_list = list(range(self.task_cnt))

            if self.task_cnt > 1:
                """MultiTask"""
                appData.setValue("batch", True)

            interval_time = time.time()
            try:
                for i, command_data in enumerate(command_list):
                    input_path, command = command_data
                    if i not in self.task_list:
                        logger.info(f"Skip task {i}")
                        continue
                    if not len(command):
                        logger.warning(f"At task {i}, Invalid Input Path: {command}")
                        continue
                    if self.kill:
                        logger.warning(f"Mission Queue Killed, Breaking")
                        break
                    logger.info(f"Designed Command:\n{command}")
                    proc_args = shlex.split(command)

                    startupinfo = sp.STARTUPINFO()
                    startupinfo.dwFlags = sp.CREATE_NEW_CONSOLE | sp.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = sp.SW_HIDE
                    self.current_proc = sp.Popen(args=proc_args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='utf-8',
                                                 errors='ignore',
                                                 universal_newlines=True, startupinfo=startupinfo)

                    flush_lines = ""
                    while self.current_proc.poll() is None:
                        if self.kill:
                            break

                        if self.pause:
                            pid = self.current_proc.pid
                            pause = psutil.Process(pid)  # 传入子进程的pid
                            pause.suspend()  # 暂停子进程
                            _msg = _translate('', '补帧已被手动暂停')
                            self.update_status(False, notice=f"\n\nWARNING, {_msg}", returncode=0)

                            while True:
                                if self.kill:
                                    break
                                elif not self.pause:
                                    pause.resume()
                                    _msg = _translate('', '补帧已继续')
                                    self.update_status(False, notice=f"\n\nWARNING, {_msg}",
                                                       returncode=0)
                                    break
                                time.sleep(0.2)
                        else:
                            line = self.current_proc.stdout.readline()
                            self.current_proc.stdout.flush()

                            """Replace Field"""
                            flush_lines += line.replace("[A", "")

                            if "error" in flush_lines.lower():
                                """Imediately Upload"""
                                logger.error(f"[In ONE LINE SHOT]: f{flush_lines}")
                                self.update_status(False, sp_status=f"{flush_lines}")
                                self.main_error = flush_lines
                                flush_lines = ""
                            elif len(flush_lines) and time.time() - interval_time > 0.1:
                                interval_time = time.time()
                                self.update_status(False, sp_status=f"{flush_lines}")
                                flush_lines = ""

                    self.update_status(False, sp_status=f"{flush_lines}")  # emit last possible infos

                    self.current_step += 1
                    _msg = _translate('', '完成')
                    self.update_status(False,
                                       f"\nINFO - {datetime.datetime.now()} {input_path} {_msg}\n\n")
                    self.maintain_multitask()
                    if appPref.value("is_rude_exit", False, type=bool):
                        Tools.kill_svfi_related()
            except Exception:
                logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))

            if self.current_proc is None:
                """Not one single task ever started"""
                logger.error("Task List Empty, Please Check Your Settings! (input fps for example)")
                _msg = _translate('', '请点击要进行的任务条目以更新设置，并确认输入输出帧率不为0后再点击补帧按钮')
                self.update_status(True, f"\nTask List is Empty!\n{_msg}",
                                   returncode=404)
                return

            self.update_status(True, returncode=self.current_proc.returncode)
            logger.info("Tasks Finished")
            if self.current_proc.returncode == 0:
                """Finish Normally"""
                if appData.value("after_mission", type=int) == 1:
                    logger.info("Task Finished Normally, User Request to Shutdown")
                    pp = Tools.popen("shutdown -s -t 120")
                    pp.wait()
                elif appData.value("after_mission", type=int) == 2:
                    logger.info("Task Finished Normally, User Request to Hibernate")
                    pp = Tools.popen("shutdown -h")
                    pp.wait()
            return
        except Exception:
            logger.error("Task Badly Finished", traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.update_status(True, returncode=1)

    def kill_proc_exec(self):
        self.kill = True
        self.current_step = len(self.task_list) if self.task_list is not None else 0
        if self.current_proc is not None:
            self.current_proc.terminate()
            _msg = _translate('', '补帧已被强制结束')
            self.update_status(False, notice=f"\n\nWARNING, {_msg}", returncode=-1)
            logger.info("Kill Process")
        else:
            logger.warning("There's no Process to kill")

    def pause_proc_exec(self):
        self.pause = not self.pause
        if self.pause:
            logger.info("Pause Process Command Fired")
        else:
            logger.info("Resume Process Command Fired")

    def get_main_error(self):
        return self.main_error

    pass


class UiBackend(QMainWindow, SVFI_UI.Ui_MainWindow):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        """
        SVFI 主界面类初始化方法

        传参变量命名手册
        ;字符串或数值：类_功能或属性
        ;属性布尔：is_类_功能
        ;使能布尔：use_类_功能
        ;特殊布尔（单一成类）：类
        添加功能三步走：
        ！！！初始化用户选项载入->将现有界面的选项缓存至配置文件中->特殊配置！！！

        添加新选项/变量 3/3 实现先于load_current_settings的特殊新配置
        :param parent:
        """
        super(UiBackend, self).__init__()
        self.setupUi(self)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)

        self.rife_thread = None
        self.chores_thread = None
        self.version = ArgumentManager.version_tag
        self.is_free = ArgumentManager.is_free
        self.is_steam = ArgumentManager.is_steam

        if appData.value("ffmpeg", "") != "ffmpeg":
            self.ffmpeg = f'"{os.path.join(appData.value("ffmpeg", ""), "ffmpeg.exe")}"'
        else:
            self.ffmpeg = appData.value("ffmpeg", "")

        if os.path.exists(appDataPath):
            logger.info("Previous Settings Found")

        self.check_gpu = False  # 是否检查过gpu
        self.current_failed = False  # 当前任务失败flag
        self.pause = False  # 当前任务是否暂停
        self.last_item = None  # 上一次点击的条目

        """Preference Maintainer"""
        self.rife_cuda_cnt = 0
        self.SVFI_Preference_form = None
        self.resize_exp = 0

        """Initiate and Check GPU"""
        self.hasNVIDIA = True
        self.settings_update_pack()

        """Initiate Beautiful Layout and Signals"""
        self.AdvanceSettingsArea.setVisible(False)
        self.ProgressBarVisibleControl.setVisible(False)
        self.settings_link_shortcut()
        self.settings_windows_ontop()

        """Link InputFileName Event"""
        self.function_enable_inputfilename_connection()

        """Dilapidation and Free Version Maintainer"""
        self.settings_free_hide()
        self.settings_dilapidation_hide()
        self.settings_load_settings_templates()

        self.STEAM = SteamUtils(self.is_steam, logger=logger)
        if self.is_steam:
            if not self.STEAM.steam_valid:
                warning_title = _translate('', "Steam认证出错！SVFI用不了啦！")
                error = self.STEAM.steam_error
                logger.error(f"Steam Validation failed\n{error}")
                self.function_send_msg(warning_title, error)
            else:
                valid_response = self.STEAM.CheckSteamAuth()
                # debug
                # valid_response = 1
                if valid_response != 0:
                    self.STEAM.steam_valid = False
                    warning_title = _translate('', "Steam认证失败！SVFI用不了啦！")
                    warning_code_msg = _translate('', '错误代码：')
                    warning_msg = f"{warning_code_msg}{valid_response}"
                    _bpg_msg = _translate('', '白嫖怪爬呀！')
                    if valid_response == 1:
                        warning_msg = f"Ticket is not valid.\n{_bpg_msg}"
                    elif valid_response == 2:
                        warning_msg = "A ticket has already been submitted for this steamID"
                    elif valid_response == 3:
                        warning_msg = "Ticket is from an incompatible interface version"
                    elif valid_response == 4:
                        warning_msg = f"Ticket is not for this game\n{_bpg_msg}"
                    elif valid_response == 5:
                        _expired_msg = _translate('', '购买的凭证过期')
                        warning_msg = f"Ticket has expired\n{_expired_msg}"
                    self.function_send_msg(warning_title, warning_msg)
                    return

                if not self.is_free:
                    valid_response = self.STEAM.CheckProDLC(0)
                    if not valid_response:
                        self.STEAM.steam_valid = False
                        warning_title = _translate('', "未购买专业版！SVFI用不了啦！")
                        warning_msg = _translate('', "请确保专业版DLC已安装")
                        self.function_send_msg(warning_title, warning_msg)
                        return

        os.chdir(appDir)
        self.function_check_read_tutorial()

    def settings_change_lang(self, lang: str):
        logger.debug(f"Translate To Lang = {lang}")
        translator.change_lang(lang)
        _app = QApplication.instance()  # 获取app实例
        _app.installTranslator(translator)  # 重新翻译主界面
        self.retranslateUi(self)

    def settings_dilapidation_hide(self):
        """Hide Dilapidated Options"""
        self.ScdetModeLabel.setVisible(False)
        self.ScdetMode.setVisible(False)
        self.ScdetFlowLen.setVisible(False)
        self.SaveCurrentSettings.setVisible(False)
        self.LoadCurrentSettings.setVisible(False)
        self.ShortCutGroup.setVisible(False)
        self.LockWHChecker.setVisible(False)
        self.AutoInterpScaleReminder.setVisible(False)
        self.AutoInterpScalePredictSizeSelector.setVisible(False)
        self.AiSrMode.setVisible(False)
        self.SrModeLabel.setVisible(False)

    def settings_free_hide(self):
        """
        ST only
        :return:
        """
        if not self.is_free:
            """Professional Version"""
            help_txt = self.OutputGuideLabel.text()
            help_txt = help_txt.replace(str(ArgumentManager.community_qq), str(ArgumentManager.professional_qq))
            self.OutputGuideLabel.setText(help_txt)
            return
        self.DupFramesTSelector.setVisible(False)
        self.DupFramesTSelector.setValue(0.2)
        self.DupRmMode.clear()
        ST_RmMode = [_translate('', "不去除重复帧"),
                     _translate('', "单一识别")]
        for m in ST_RmMode:
            self.DupRmMode.addItem(m)

        ST_HwaccelMode = ['CPU', 'QSV']
        try:
            self.HwaccelSelector.disconnect()
        except:
            pass
        self.HwaccelSelector.clear()
        for m in ST_HwaccelMode:
            self.HwaccelSelector.addItem(m)
        self.HwaccelSelector.currentTextChanged.connect(self.on_HwaccelSelector_currentTextChanged)

        ST_HdrMode = ['Auto', 'None']
        self.HDRModeSelector.clear()
        for m in ST_HdrMode:
            self.HDRModeSelector.addItem(m)

        self.StartPoint.setVisible(False)
        self.EndPoint.setVisible(False)
        self.StartPointLabel.setVisible(False)
        self.EndPointLabel.setVisible(False)
        self.ScdetOutput.setVisible(False)
        self.ScdetUseMix.setVisible(False)
        self.UseAiSR.setChecked(False)
        self.UseAiSR.setEnabled(False)
        self.SrField.setVisible(False)
        self.RenderSettingsLabel.setVisible(False)
        self.RenderSettingsGroup.setVisible(False)
        self.UseMultiCardsChecker.setVisible(False)
        self.TtaModeSelector.setVisible(False)
        self.TtaIterTimesSelector.setVisible(False)
        self.TtaModeLabel.setVisible(False)
        self.EvictFlickerChecker.setVisible(False)
        self.AutoInterpScaleChecker.setVisible(False)
        self.ReverseChecker.setVisible(False)
        self.ProAdLabel_1.setVisible(True)

        self.DeinterlaceChecker.setVisible(False)
        self.FastDenoiseChecker.setVisible(False)
        self.HwaccelDecode.setVisible(False)
        self.EncodeThreadField.setVisible(False)
        self.HwaccelPresetLabel.setVisible(False)
        self.HwaccelPresetSelector.setVisible(False)

        self.GifBox.setEnabled(False)
        self.RenderBox.setEnabled(False)
        self.ExtractBox.setEnabled(False)
        self.SettingsPresetBox.setEnabled(False)

        self.DebugChecker.setVisible(False)

    def settings_update_pack(self, item_update=False, template_update=False):
        self.settings_initiation(item_update=item_update, template_update=False)
        self.settings_update_gpu_info(item_update=item_update)  # Flush GPU Info, 1
        self.on_UseNCNNButton_clicked(silent=True)  # G2
        self.settings_update_rife_model_info()
        self.on_HwaccelSelector_currentTextChanged()  # Flush Encoder Sets, 1
        self.on_EncoderSelector_currentTextChanged()  # E2
        self.on_UseEncodeThread_clicked()  # E3
        self.on_slowmotion_clicked()
        self.on_MBufferChecker_clicked()
        self.on_DupRmMode_currentTextChanged()
        self.on_ScedetChecker_clicked()
        self.on_ImgOutputChecker_clicked()
        self.on_AutoInterpScaleChecker_clicked()
        self.on_UseMultiCardsChecker_clicked()
        self.on_InterpExpReminder_toggled()
        self.on_UseAiSR_clicked()
        self.on_AiSrSelector_currentTextChanged()
        self.on_ResizeTemplate_currentTextChanged()
        self.on_TtaModeSelector_currentTextChanged()
        self.on_ExpertMode_changed()
        self.settings_initiation(item_update=item_update, template_update=False)
        pass

    def settings_initiation(self, item_update=False, template_update=False):
        """
        初始化用户选项载入
        从配置文件中读取上一次设置并初始化页面
        添加新选项/变量 1/3 appData -> Options
        :item_update: if inputs' current item changed, activate this
        :return:
        """
        global appData

        if not item_update:
            """New Initiation of GUI"""
            try:
                input_list_data = json.loads(appPref.value("gui_inputs", ""))
                if not self.InputFileName.count():
                    for item_data in input_list_data['inputs']:
                        config_maintainer = SVFI_Config_Manager(item_data, appDir)
                        input_path = config_maintainer.FetchConfig()
                        if input_path is not None and os.path.exists(input_path):
                            self.InputFileName.addFileItem(item_data['input_path'],
                                                           item_data['task_id'])  # resume previous tasks
            except json.decoder.JSONDecodeError:
                logger.error("Could Not Find Valid GUI Inputs from appData, leave blank")

            """Maintain SVFI Startup Resolution"""
            desktop = QApplication.desktop()
            pos = appData.value("pos", QVariant(QPoint(960, 540)))
            size = appData.value("size", QVariant(QSize(int(desktop.width() * 0.8), int(desktop.height() * 0.8))))
            self.resize(size)
            self.move(pos)

        if not template_update:
            """Basic Configuration"""
            self.OutputFolder.setText(appData.value("output_dir"))
            self.InputFPS.setText(appData.value("input_fps", "0"))
            self.OutputFPS.setText(appData.value("target_fps", ""))
            self.OutputFPSReminder.setChecked(not appData.value("is_exp_prior", False, type=bool))
            self.InterpExpReminder.setChecked(appData.value("is_exp_prior", True, type=bool))
            self.ExpSelecter.setCurrentText("x" + str(2 ** int(appData.value("rife_exp", "1"))))
            self.ExtSelector.setCurrentText(appData.value("output_ext", "mp4"))
            self.ImgOutputChecker.setChecked(appData.value("is_img_output", False, type=bool))
            appData.setValue("is_img_input", appData.value("is_img_input", False))
            self.KeepChunksChecker.setChecked(not appData.value("is_output_only", True, type=bool))
            self.StartPoint.setTime(QTime.fromString(appData.value("input_start_point", "00:00:00"), "HH:mm:ss"))
            self.EndPoint.setTime(QTime.fromString(appData.value("input_end_point", "00:00:00"), "HH:mm:ss"))
            self.StartChunk.setValue(appData.value("output_chunk_cnt", -1, type=int))
            self.StartFrame.setValue(appData.value("interp_start", -1, type=int))
            self.ResumeRiskChecker.setChecked(appData.value("risk_resume_mode", True, type=bool))

        self.DebugChecker.setChecked(appData.value("debug", False, type=bool))

        """Output Resize Configuration"""
        self.CropHeightSettings.setValue(appData.value("crop_height", 0, type=int))
        self.CropWidthpSettings.setValue(appData.value("crop_width", 0, type=int))
        self.ResizeHeightSettings.setValue(appData.value("resize_height", 0, type=int))
        self.ResizeWidthSettings.setValue(appData.value("resize_width", 0, type=int))
        self.ResizeTemplate.setCurrentIndex(appData.value("resize_settings_index", 0, type=int))

        """Render Configuration"""
        self.UseCRF.setChecked(appData.value("use_crf", True, type=bool))
        self.CRFSelector.setValue(appData.value("render_crf", 16, type=int))
        self.UseTargetBitrate.setChecked(appData.value("use_bitrate", False, type=bool))
        self.BitrateSelector.setValue(appData.value("render_bitrate", 90, type=float))
        self.HwaccelSelector.setCurrentText(appData.value("render_hwaccel_mode", "CPU", type=str))
        self.HwaccelPresetSelector.setCurrentText(appData.value("render_hwaccel_preset", "None"))
        self.HwaccelDecode.setChecked(appData.value("use_hwaccel_decode", True, type=bool))
        self.UseEncodeThread.setChecked(appData.value("use_manual_encode_thread", False, type=bool))
        self.EncodeThreadSelector.setValue(appData.value("render_encode_thread", 16, type=int))
        self.EncoderSelector.setCurrentText(appData.value("render_encoder", "H264,8bit"))
        self.PresetSelector.setCurrentText(appData.value("render_encoder_preset", "slow"))
        self.HDRModeSelector.setCurrentIndex(appData.value("hdr_mode", 0, type=int))
        self.FFmpegCustomer.setText(appData.value("render_ffmpeg_customized", ""))
        self.ExtSelector.setCurrentText(appData.value("output_ext", "mp4"))
        self.RenderGapSelector.setValue(appData.value("render_gap", 1000, type=int))
        self.SaveAudioChecker.setChecked(appData.value("is_save_audio", True, type=bool))
        self.FastDenoiseChecker.setChecked(appData.value("use_fast_denoise", False, type=bool))
        self.HDRModeSelector.setCurrentIndex(appData.value("hdr_mode", 0, type=int))
        self.QuickExtractChecker.setChecked(appData.value("is_quick_extract", True, type=bool))
        self.DeinterlaceChecker.setChecked(appData.value("use_deinterlace", False, type=bool))

        """Slowmotion Configuration"""
        self.slowmotion.setChecked(appData.value("is_render_slow_motion", False, type=bool))
        self.SlowmotionFPS.setText(appData.value("render_slow_motion_fps", "", type=str))
        self.GifLoopChecker.setChecked(appData.value("gif_loop", True, type=bool))

        """Scdet and RD Configuration"""
        self.ScedetChecker.setChecked(not appData.value("is_no_scdet", False, type=bool))
        self.ScdetSelector.setValue(appData.value("scdet_threshold", 12, type=int))
        self.ScdetUseMix.setChecked(appData.value("is_scdet_mix", False, type=bool))
        self.ScdetOutput.setChecked(appData.value("is_scdet_output", False, type=bool))
        self.ScdetFlowLen.setCurrentIndex(appData.value("scdet_flow_cnt", 0, type=int))
        self.UseFixedScdet.setChecked(appData.value("use_scdet_fixed", False, type=bool))
        self.ScdetMaxDiffSelector.setValue(appData.value("scdet_fixed_max", 40, type=int))
        self.ScdetMode.setCurrentIndex(appData.value("scdet_mode", 0, type=int))
        self.DupRmMode.setCurrentIndex(appData.value("remove_dup_mode", 0, type=int))
        self.UseSobelChecker.setChecked(appData.value("use_dedup_sobel", False, type=bool))
        self.DupFramesTSelector.setValue(appData.value("remove_dup_threshold", 10.00, type=float))

        """AI Super Resolution Configuration"""
        self.UseAiSR.setChecked(appData.value("use_sr", False, type=bool))
        self.SrTileSizeSelector.setValue(appData.value("sr_tilesize", 100, type=int))
        self.AiSrSelector.setCurrentText(appData.value("use_sr_algo", "realESR"))
        last_sr_model = appData.value("use_sr_model", "")
        if len(last_sr_model):
            self.AiSrModuleSelector.setCurrentText(last_sr_model)
        else:
            self.on_UseAiSR_clicked()
        self.AiSrMode.setCurrentIndex(appData.value("use_sr_mode", 0, type=int))

        """RIFE Configuration"""
        self.FP16Checker.setChecked(appData.value("use_rife_fp16", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("rife_scale", "1.00"))
        self.ReverseChecker.setChecked(appData.value("is_rife_reverse", False, type=bool))
        self.ForwardEnsembleChecker.setChecked(appData.value("use_rife_forward_ensemble", False, type=bool))
        self.AutoInterpScaleChecker.setChecked(appData.value("use_rife_auto_scale", False, type=bool))
        self.AutoInterpScalePredictSizeSelector.setValue(appData.value("rife_auto_scale_predict_size", 64, type=int))
        self.on_AutoInterpScaleChecker_clicked()
        self.UseNCNNButton.setChecked(appData.value("use_ncnn", False, type=bool))
        self.EvictFlickerChecker.setChecked(appData.value("use_evict_flicker", False, type=bool))
        self.TtaModeSelector.setCurrentIndex(appData.value("rife_tta_mode", 0, type=int))
        self.TtaIterTimesSelector.setValue(appData.value("rife_tta_iter", 1, type=int))
        self.ncnnInterpThreadCnt.setValue(appData.value("ncnn_thread", 4, type=int))
        self.ncnnSelectGPU.setValue(appData.value("ncnn_gpu", 0, type=int))
        self.UseMultiCardsChecker.setChecked(appData.value("use_rife_multi_cards", False, type=bool))

        # Update RIFE Model
        rife_model_list = []
        for item_data in range(self.ModuleSelector.count()):
            rife_model_list.append(self.ModuleSelector.itemText(item_data))
        if appData.value("rife_model_name", "") in rife_model_list:
            self.ModuleSelector.setCurrentText(appData.value("rife_model_name", ""))

        """REM Management Configuration"""
        self.MBufferChecker.setChecked(appData.value("use_manual_buffer", False, type=bool))
        self.BufferSizeSelector.setValue(appData.value("manual_buffer_size", 1, type=int))

        # self.setAttribute(Qt.WA_TranslucentBackground)

    def settings_load_current(self):
        """
        将现有界面的选项缓存至配置文件中
        添加新选项/变量 2/3 Options -> appData
        :return:
        """
        global ols_potential

        appData.setValue("app_dir", appDir)
        if not os.path.exists(ols_potential):
            ols_potential = r"D:\60-fps-Project\Projects\RIFE GUI\one_line_shot_args.py"
            logger.info("Change to Debug Path")

        """Load Inputs"""
        appPref.setValue("gui_inputs", self.InputFileName.saveTasks())

        """Input Basic Input Information"""
        appData.setValue("version", self.version)
        appData.setValue("output_dir", self.OutputFolder.text())
        appData.setValue("input_fps", self.InputFPS.text())
        appData.setValue("target_fps", self.OutputFPS.text())
        appData.setValue("is_exp_prior", self.InterpExpReminder.isChecked())
        appData.setValue("rife_exp", int(math.log(int(self.ExpSelecter.currentText()[1:]), 2)))
        appData.setValue("is_img_output", self.ImgOutputChecker.isChecked())
        appData.setValue("is_output_only", not self.KeepChunksChecker.isChecked())
        appData.setValue("is_save_audio", self.SaveAudioChecker.isChecked())
        appData.setValue("output_ext", self.ExtSelector.currentText())
        appData.setValue("risk_resume_mode", self.ResumeRiskChecker.isChecked())

        """Input Time Stamp"""
        appData.setValue("input_start_point", self.StartPoint.time().toString("HH:mm:ss"))
        appData.setValue("input_end_point", self.EndPoint.time().toString("HH:mm:ss"))
        appData.setValue("output_chunk_cnt", self.StartChunk.value())
        appData.setValue("interp_start", self.StartFrame.value())
        appData.setValue("render_gap", self.RenderGapSelector.value())

        """Render"""
        appData.setValue("use_crf", self.UseCRF.isChecked())
        appData.setValue("use_bitrate", self.UseTargetBitrate.isChecked())
        appData.setValue("render_crf", self.CRFSelector.value())
        appData.setValue("render_bitrate", self.BitrateSelector.value())
        appData.setValue("render_encoder_preset", self.PresetSelector.currentText())
        appData.setValue("render_encoder", self.EncoderSelector.currentText())
        appData.setValue("render_hwaccel_mode", self.HwaccelSelector.currentText())
        appData.setValue("render_hwaccel_preset", self.HwaccelPresetSelector.currentText())
        appData.setValue("use_hwaccel_decode", self.HwaccelDecode.isChecked())
        appData.setValue("use_manual_encode_thread", self.UseEncodeThread.isChecked())
        appData.setValue("render_encode_thread", self.EncodeThreadSelector.value())
        appData.setValue("is_quick_extract", self.QuickExtractChecker.isChecked())
        appData.setValue("hdr_mode", self.HDRModeSelector.currentIndex())
        appData.setValue("render_ffmpeg_customized", self.FFmpegCustomer.text())
        appData.setValue("no_concat", False)  # always concat
        appData.setValue("use_fast_denoise", self.FastDenoiseChecker.isChecked())

        """Special Render Effect"""
        appData.setValue("gif_loop", self.GifLoopChecker.isChecked())
        appData.setValue("is_render_slow_motion", self.slowmotion.isChecked())
        appData.setValue("render_slow_motion_fps", self.SlowmotionFPS.text())
        if appData.value("is_render_slow_motion", False, type=bool):
            appData.setValue("is_save_audio", False)
            self.SaveAudioChecker.setChecked(False)
        appData.setValue("use_deinterlace", self.DeinterlaceChecker.isChecked())

        appData.setValue("resize_settings_index", self.ResizeTemplate.currentIndex())
        height, width = self.ResizeHeightSettings.value(), self.ResizeWidthSettings.value()
        appData.setValue("resize_width", width)
        appData.setValue("resize_height", height)
        if all((width, height)):
            appData.setValue("resize", f"{width}x{height}")
        else:
            appData.setValue("resize", f"")
        width, height = self.CropWidthpSettings.value(), self.CropHeightSettings.value()
        appData.setValue("crop_width", width)
        appData.setValue("crop_height", height)
        if any((width, height)):
            appData.setValue("crop", f"{width}:{height}")
        else:
            appData.setValue("crop", f"")

        """Scene Detection"""
        appData.setValue("is_no_scdet", not self.ScedetChecker.isChecked())
        appData.setValue("is_scdet_mix", self.ScdetUseMix.isChecked())
        appData.setValue("use_scdet_fixed", self.UseFixedScdet.isChecked())
        appData.setValue("is_scdet_output", self.ScdetOutput.isChecked())
        appData.setValue("scdet_threshold", self.ScdetSelector.value())
        appData.setValue("scdet_fixed_max", self.ScdetMaxDiffSelector.value())
        appData.setValue("scdet_flow_cnt", self.ScdetFlowLen.currentIndex())
        appData.setValue("scdet_mode", self.ScdetMode.currentIndex())

        """Duplicate Frames Removal"""
        appData.setValue("remove_dup_mode", self.DupRmMode.currentIndex())
        appData.setValue("use_dedup_sobel", self.UseSobelChecker.isChecked())
        appData.setValue("remove_dup_threshold", self.DupFramesTSelector.value())

        """RAM Management"""
        appData.setValue("use_manual_buffer", self.MBufferChecker.isChecked())
        appData.setValue("manual_buffer_size", self.BufferSizeSelector.value())

        """Super Resolution Settings"""
        appData.setValue("use_sr", self.UseAiSR.isChecked())
        appData.setValue("use_sr_algo", self.AiSrSelector.currentText())
        appData.setValue("use_sr_model", self.AiSrModuleSelector.currentText())
        appData.setValue("use_sr_mode", self.AiSrMode.currentIndex())
        appData.setValue("sr_tilesize", self.SrTileSizeSelector.value())
        appData.setValue("resize_exp", self.resize_exp)

        """RIFE Settings"""
        appData.setValue("use_ncnn", self.UseNCNNButton.isChecked())
        appData.setValue("ncnn_thread", self.ncnnInterpThreadCnt.value())
        appData.setValue("ncnn_gpu", self.ncnnSelectGPU.value())
        appData.setValue("rife_tta_mode", self.TtaModeSelector.currentIndex())
        appData.setValue("rife_tta_iter", self.TtaIterTimesSelector.value())
        appData.setValue("use_evict_flicker", self.EvictFlickerChecker.isChecked())
        appData.setValue("use_rife_fp16", self.FP16Checker.isChecked())
        appData.setValue("rife_scale", self.InterpScaleSelector.currentText())
        appData.setValue("is_rife_reverse", self.ReverseChecker.isChecked())
        appData.setValue("rife_model",
                         os.path.join(appData.value("rife_model_dir", ""), self.ModuleSelector.currentText()))
        appData.setValue("rife_model_name", self.ModuleSelector.currentText())
        appData.setValue("rife_cuda_cnt", self.rife_cuda_cnt)
        appData.setValue("use_rife_multi_cards", self.UseMultiCardsChecker.isChecked())
        appData.setValue("use_specific_gpu", self.DiscreteCardSelector.currentIndex())
        appData.setValue("use_rife_auto_scale", self.AutoInterpScaleChecker.isChecked())
        appData.setValue("rife_auto_scale_predict_size", self.AutoInterpScalePredictSizeSelector.value())
        appData.setValue("use_rife_forward_ensemble", self.ForwardEnsembleChecker.isChecked())

        """Debug Mode"""
        appData.setValue("debug", self.DebugChecker.isChecked())

        """Preferences"""
        for pref_key in appPref.allKeys():
            appData.setValue(pref_key, appPref.value(pref_key))

        """SVFI Main Page Position and Size"""
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))

        logger.info("[Main]: Download all settings")
        self.OptionCheck.isReadOnly = True
        appData.sync()
        appPref.sync()
        try:
            if not os.path.samefile(appData.fileName(), appDataPath):
                shutil.copy(appData.fileName(), appDataPath)
        except FileNotFoundError:
            logger.info("Unable to save Configs, probably permanent loss")
        pass

    def settings_check_args(self) -> bool:
        """
        Check are all args available
        :return:
        """
        input_paths = self.function_get_input_paths()
        output_dir = self.OutputFolder.text()

        if not len(input_paths):
            self.function_send_msg("Empty Input", _translate('', "请输入要补帧的文件和输出文件夹"))
            return False

        if not len(output_dir):
            self.OutputFolder.setText(os.path.dirname(input_paths[0]))
            output_dir = self.OutputFolder.text()

        if ' ' in output_dir and '.' in output_dir:
            self.function_send_msg("Invalid Output Folder", _translate('', "输出文件夹同时存在空格和'.'，请删除路径空格"))
            return False

        if Tools.check_non_ascii(output_dir):
            reply = self.function_send_msg("Non ASCII Detected",
                                           _translate('', "输出路径存在非英文字符，这可能导致程序异常\n(不支持Dolby Vision)\n是否继续？"), 4)
            if reply == QMessageBox.No:
                return False

        if len(input_paths) > 1:
            self.ProgressBarVisibleControl.setVisible(True)
        else:
            self.ProgressBarVisibleControl.setVisible(False)

        if not os.path.exists(output_dir):
            logger.info("Not Exists OutputFolder")
            self.function_send_msg("Output Folder Not Found", _translate('', "输入文件或输出文件夹不存在！请确认输入"))
            return False

        if os.path.isfile(output_dir):
            """Auto set Output Dir to correct form"""
            self.OutputFolder.setText(os.path.dirname(output_dir))

        for path in input_paths:
            if not os.path.exists(path):
                logger.info(f"Not Exists Input Source: {path}")
                _msg1 = _translate('', '输入文件:')
                _msg2 = _translate('', '不存在！请确认输入!')
                self.function_send_msg("Input Source Not Found", f"{_msg1}\n{path}\n{_msg2}")
                return False

        try:
            float(self.InputFPS.text())
            float(self.OutputFPS.text())
        except Exception:
            self.function_send_msg("Wrong Inputs", _translate('', "请确认输入和输出帧率为有效数据"))
            return False

        try:
            if self.slowmotion.isChecked():
                float(self.SlowmotionFPS.text())
        except Exception:
            self.function_send_msg("Wrong Inputs", _translate('', "请确认慢动作输入帧率"))
            return False

        return True

    def settings_set_start_info(self, start_frame, start_chunk, custom_prior=False):
        """
        设置启动帧数和区块信息
        :param custom_prior: Input is priority
        :param start_frame: StartFrame
        :param start_chunk: StartChunk
        :return: False: Custom Parameters Detected, no chunk removal is needed
        """
        if custom_prior:
            if self.StartFrame.value() != 0 and self.StartChunk != 1:
                return False
        self.StartFrame.setValue(start_frame)
        self.StartChunk.setValue(start_chunk)
        return True

    def settings_auto_set(self):
        """
        自动根据现有区块设置启动信息
        :return:
        """
        if not self.settings_check_args():
            return
        if not self.InputFileName.count():
            return
        current_item = self.InputFileName.item(self.InputFileName.currentRow())
        if current_item is None:
            self.function_send_msg(f"恢复进度？", _translate('', "正在使用队列的第一个任务进行进度检测"))
            self.InputFileName.setCurrentRow(0)
            current_item = self.InputFileName.currentItem()
        output_dir = self.OutputFolder.text()

        widget_data = self.InputFileName.getWidgetData(current_item)
        input_path = widget_data.get('input_path')
        task_id = widget_data.get('task_id')
        project_dir = os.path.join(output_dir,
                                   f"{Tools.get_filename(input_path)}_{task_id}")
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
            _msg1 = _translate('', '未找到与第')
            _msg2 = _translate('', '个任务相关的进度信息')
            self.function_send_msg(f"Failed to Resume Workflow", f"{_msg1}{widget_data['row'] + 1}{_msg2}", 3)
            self.settings_set_start_info(0, 1, False)  # start from zero
            return

        if self.ImgOutputChecker.isChecked():
            """Img Output"""
            img_io = ImgSeqIO(logger=logger, folder=project_dir, is_tool=True, output_ext=self.ExtSelector.currentText())
            last_img = img_io.get_write_start_frame()  # output_dir
            if last_img:
                reply = self.function_send_msg(f"Resume Workflow?", _translate('', "检测到未完成的图片序列补帧任务，载入进度？"), 3)
                if reply == QMessageBox.No:
                    self.settings_set_start_info(0, 1, False)  # start from zero
                    logger.info("User Abort Auto Set")
                    return
                self.settings_set_start_info(int(last_img), 1, False)
            else:
                _msg1 = _translate('', '未找到与第')
                _msg2 = _translate('', '个任务相关的进度信息')
                self.function_send_msg(f"Failed to Resume Workflow", f"{_msg1}{widget_data['row'] + 1}{_msg2}", 3)
                self.settings_set_start_info(-1, -1, False)
            return

        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(project_dir)
        if not len(chunk_paths):
            _msg1 = _translate('', '未找到与第')
            _msg2 = _translate('', '个任务相关的进度信息')
            self.function_send_msg(f"Failed to Resume Workflow", f"{_msg1}{widget_data['row'] + 1}{_msg2}", 3)
            logger.info("AutoSet find None to resume interpolation")
            self.settings_set_start_info(0, 1, False)
            return

        if chunk_cnt > 0:
            reply = self.function_send_msg(f"Resume Workflow?", _translate('', "检测到未完成的补帧任务，载入进度？"), 3)
            if reply == QMessageBox.No:
                self.settings_set_start_info(0, 1, False)
                logger.info("User Abort Auto Set")
                return
        self.settings_set_start_info(last_frame + 1, chunk_cnt + 1, False)
        return

    def settings_load_config(self, config_path: str):
        """

        :param config_path:
        :return:
        """
        global appData
        appData = QSettings(config_path, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")

    def settings_maintain_item_settings(self, widget_data: dict):
        global appData
        use_global_settings = appPref.value("use_global_settings", False, type=bool)
        if use_global_settings:
            """First detect using use global settings"""
            self.settings_load_config(appDataPath)  # change to root settings
        self.settings_load_current()  # 保存跳转前设置
        initiated = True
        if self.last_item is None:
            initiated = False
            self.last_item = widget_data
        config_maintainer = SVFI_Config_Manager(self.last_item, appDir)
        if initiated:
            # 仅在初始化的第一个任务时
            config_maintainer.DuplicateConfig()  # 将当前设置保存到上一任务的配置文件，并准备跳转到新任务
        if not use_global_settings:
            """使用已存在的（上一）任务配置文件载入设置"""
            config_maintainer = SVFI_Config_Manager(widget_data, appDir)
            config_path = config_maintainer.FetchConfig()
            if config_path is None:
                config_maintainer.DuplicateConfig()  # 利用当前系统全局设置保存当前任务配置
                # config_path = config_maintainer.FetchConfig()
            config_maintainer.UpdateRootConfig()
            self.settings_load_config(appDataPath)
            self.settings_update_pack(item_update=not use_global_settings)
        self.last_item = widget_data

    @pyqtSlot(bool)
    def settings_load_settings_templates(self):
        """
        读取存在的自定义预设
        :return:
        """
        config_dir = os.path.join(appDir, 'Configs', "SVFI_Config_Template_*.ini")
        template_paths = glob.glob(config_dir)
        self.SettingsTemplateSelector.clear()
        for tp in template_paths:
            template_name = re.findall('SVFI_Config_Template_(.*?)\.ini', tp)
            if len(template_name) and "Presets" not in template_name:
                self.SettingsTemplateSelector.addItem(template_name[0])

    def settings_update_gpu_info(self, item_update=False):
        if item_update:
            card_id = appData.value("use_specific_gpu", 0, type=int)
            if self.DiscreteCardSelector.count() - 1 >= card_id:
                self.DiscreteCardSelector.setCurrentIndex(card_id)
            return
        cuda_infos = {}
        self.rife_cuda_cnt = torch.cuda.device_count()
        for i in range(self.rife_cuda_cnt):
            card = torch.cuda.get_device_properties(i)
            info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
            cuda_infos[f"{i}"] = info
        logger.debug(f"NVIDIA data: {cuda_infos}")

        if not len(cuda_infos):
            self.hasNVIDIA = False
            self.function_send_msg("No NVIDIA Card Found", _translate('', "未找到N卡，将使用A卡或核显"))
            appData.setValue("use_ncnn", True)
            self.UseNCNNButton.setChecked(True)
            self.UseNCNNButton.setEnabled(False)
            self.on_UseNCNNButton_clicked()
            return
        else:
            appData.setValue("use_ncnn", self.UseNCNNButton.isChecked())

        self.DiscreteCardSelector.clear()
        for gpu in cuda_infos:
            self.DiscreteCardSelector.addItem(f"{gpu}: {cuda_infos[gpu]}")
        self.check_gpu = True
        return cuda_infos

    def settings_update_rife_model_info(self):
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ncnn_dir = os.path.join(app_dir, "ncnn")
        rife_ncnn_dir = os.path.join(ncnn_dir, "rife")
        if self.UseNCNNButton.isChecked():
            model_dir = os.path.join(rife_ncnn_dir, "models")
        else:
            model_dir = os.path.join(app_dir, "train_log")
        appData.setValue("rife_model_dir", model_dir)

        if not os.path.exists(model_dir):
            logger.info(f"Not find Module dir at {model_dir}")
            self.function_send_msg("Model Dir Not Found", _translate('', "未找到补帧模型路径，请检查！"))
            return
        rife_model_list = list()
        for m in os.listdir(model_dir):
            if not os.path.isfile(os.path.join(model_dir, m)):
                rife_model_list.append(m)
        # rife_model_list.reverse()
        self.ModuleSelector.clear()
        for mod in rife_model_list:
            self.ModuleSelector.addItem(f"{mod}")

    def settings_update_sr_algo(self):
        sr_ncnn_dir = self.function_get_SuperResolution_paths()

        if not os.path.exists(sr_ncnn_dir):
            logger.info(f"Not find SR Algorithm dir at {sr_ncnn_dir}")
            self.function_send_msg("Model Dir Not Found", _translate('', "未找到补帧模型路径，请检查！"))
            return

        algo_list = list()
        for m in os.listdir(sr_ncnn_dir):
            if not os.path.isfile(os.path.join(sr_ncnn_dir, m)):
                algo_list.append(m)

        self.AiSrSelector.clear()
        for algo in algo_list:
            self.AiSrSelector.addItem(f"{algo}")

    def settings_update_sr_model(self):
        """
        更新NCNN超分模型
        :return:
        """
        current_sr_algo = self.AiSrSelector.currentText()
        if not len(current_sr_algo) or self.is_free or not self.UseAiSR.isChecked():
            return
        sr_algo_ncnn_dir = self.function_get_SuperResolution_paths(path_type=1,
                                                                   key_word=current_sr_algo)

        if not os.path.exists(sr_algo_ncnn_dir):
            logger.info(f"Not find SR Algorithm dir at {sr_algo_ncnn_dir}")
            self.function_send_msg("Model Dir Not Found", _translate('', "未找到超分模型，请检查！"))
            return

        model_list = list()
        for m in os.listdir(sr_algo_ncnn_dir):
            if not os.path.isfile(os.path.join(sr_algo_ncnn_dir, m)):
                model_list.append(m)
            if "realESR" in current_sr_algo:
                # pth model only
                model_list.append(m)

        self.AiSrModuleSelector.clear()
        for model in model_list:
            self.AiSrModuleSelector.addItem(f"{model}")

    def settings_link_shortcut(self):
        self.homeActionButton.setShortcut("ctrl+1")
        self.outputActionButton.setShortcut("ctrl+2")
        self.resumeActionButton.setShortcut("ctrl+3")
        self.scdetActionButton.setShortcut("ctrl+4")
        self.resolutionActionButton.setShortcut("ctrl+5")
        self.renderActionButton.setShortcut("ctrl+6")
        self.rifeActionButton.setShortcut("ctrl+7")
        self.presetActionButton.setShortcut("ctrl+8")
        self.toolboxActionButton.setShortcut("ctrl+9")
        self.homeActionButton.clicked.connect(lambda i=0: self.tabWidget.setCurrentIndex(i))
        self.outputActionButton.clicked.connect(lambda i=1: self.tabWidget.setCurrentIndex(i))
        self.resumeActionButton.clicked.connect(lambda i=0: self.toolBox.setCurrentIndex(i))
        self.scdetActionButton.clicked.connect(lambda i=1: self.toolBox.setCurrentIndex(i))
        self.resolutionActionButton.clicked.connect(lambda i=2: self.toolBox.setCurrentIndex(i))
        self.renderActionButton.clicked.connect(lambda i=3: self.toolBox.setCurrentIndex(i))
        self.rifeActionButton.clicked.connect(lambda i=4: self.toolBox.setCurrentIndex(i))
        self.presetActionButton.clicked.connect(lambda i=5: self.toolBox.setCurrentIndex(i))
        self.toolboxActionButton.clicked.connect(lambda i=6: self.toolBox.setCurrentIndex(i))

    def settings_windows_ontop(self):
        if not appPref.value("is_windows_ontop", False, type=bool):
            self.setWindowFlags(QtCore.Qt.Widget)  # 取消置顶
        else:
            self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)  # 置顶

    def function_generate_log(self, mode=0):
        """
        生成日志并提示用户
        :param mode:0 Error Log 1 Settings Log
        :return:
        """
        preview_args = UiPreviewArgsDialog(self).ArgsLabel.text()
        preview_args = html.unescape("\n".join(re.findall('">(.*?)</span>', preview_args)))
        _msg1 = _translate('', '[导出设置预览]')
        status_check = f"{_msg1}\n\n{preview_args}\n\n"
        for key in appData.allKeys():
            status_check += f"{key} => {appData.value(key)}\n"
        if mode == 0:
            _msg1 = _translate('', '[设置信息]')
            status_check += f"\n\n{_msg1}\n\n"
            status_check += self.OptionCheck.toPlainText()
            log_path = os.path.join(self.OutputFolder.text(), "log", f"{datetime.datetime.now().date()}.error.log")
        else:  # 1
            log_path = os.path.join(self.OutputFolder.text(), "log", f"{datetime.datetime.now().date()}.settings.log")

        log_path_dir = os.path.dirname(log_path)
        if not os.path.exists(log_path_dir):
            os.mkdir(log_path_dir)
        with open(log_path, "w", encoding="utf-8") as w:
            w.write(status_check)
        if not self.DebugChecker.isChecked():
            os.startfile(log_path_dir)

    def function_get_input_paths(self):
        """
        获取输入文件路径列表
        :return:
        """
        widgetres = []
        count = self.InputFileName.count()
        for i in range(count):
            try:
                widgetres.append(self.InputFileName.itemWidget(self.InputFileName.item(i)).input_path)
            except:
                pass
        return widgetres

    def function_send_msg(self, title, string, msg_type=1):
        """
        标准化输出界面提示信息

        :param title:
        :param string:
        :param msg_type: 1 warning 2 info 3 question
        :return:
        """
        if appPref.value("is_gui_quiet", False, type=bool):
            return QMessageBox.Yes
        QMessageBox.setWindowIcon(self, QIcon(ico_path))
        if msg_type == 1:
            reply = QMessageBox.warning(self,
                                        f"{title}",
                                        f"{string}",
                                        QMessageBox.Yes)
        elif msg_type == 2:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes)
        elif msg_type == 3:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes | QMessageBox.No)
        elif msg_type == 4:
            reply = QMessageBox.warning(self,
                                        f"{title}",
                                        f"{string}",
                                        QMessageBox.Yes | QMessageBox.No)

        else:
            return
        return reply

    def function_select_file(self, filename, folder=False, _filter=None, multi=False):
        """
        用户选择文件
        :param filename:
        :param folder:
        :param _filter:
        :param multi:
        :return:
        """
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption=_translate('', "选取文件夹"))
            return directory
        if multi:
            files = QFileDialog.getOpenFileNames(None, caption=f"Select {filename}", filter=_filter)
            return files[0]
        directory = QFileDialog.getOpenFileName(None, caption=f"Select {filename}", filter=_filter)
        return directory[0]

    def function_quick_concat(self):
        """
        快速合并
        :return:
        """
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.settings_load_current()
        if not input_v or not input_a or not output_v:
            self.function_send_msg("Parameters unfilled", _translate('', "请填写输入或输出视频路径！"))
            return

        ffmpeg_command = (f"{self.ffmpeg} -i {Tools.fillQuotation(input_a)} -i {Tools.fillQuotation(input_v)} "
                          f"-map 1:v:0 -map 0:a:0 -c:v copy -c:a copy "
                          f"-shortest {Tools.fillQuotation(output_v)} -y").replace("\\", "/")
        logger.info(f"[GUI] concat {ffmpeg_command}")
        self.chores_thread = UiRunThread(ffmpeg_command, data={"type": "音视频合并"})
        self.chores_thread.run_signal.connect(self.function_update_chores_finish)
        self.chores_thread.start()
        self.ConcatButton.setEnabled(False)

    def function_update_chores_finish(self, data: dict):
        mission_type = data['data']['type']
        _msg1 = _translate('', '任务完成')
        self.function_send_msg("Chores Mission", f"{mission_type}{_msg1}", msg_type=2)
        self.ConcatButton.setEnabled(True)
        self.GifButton.setEnabled(True)

    def function_quick_gif(self):
        """
        快速生成GIF
        :return:
        """
        input_v = self.GifInput.text()
        output_v = self.GifOutput.text()
        self.settings_load_current()
        if not input_v or not output_v:
            self.function_send_msg("Parameters unfilled", _translate('', "请填写输入或输出视频路径！"))
            return
        if not appData.value("target_fps"):
            appData.setValue("target_fps", 50)
            logger.info("Not find output GIF fps, Auto set GIF output fps to 50 as it's smooth enough")
        target_fps = appData.value("target_fps", 50, type=float)

        width = self.ResizeWidthSettings.value()
        height = self.ResizeHeightSettings.value()
        resize = f"scale={width}:{height},"
        if not all((width, height)):
            resize = ""
        ffmpeg_command = (f'{self.ffmpeg} -hide_banner -i {Tools.fillQuotation(input_v)} -r {target_fps} '
                          f'-lavfi "{resize}split[s0][s1];'
                          f'[s0]palettegen=stats_mode=diff[p];'
                          f'[s1][p]paletteuse=dither=floyd_steinberg" '
                          f'{"-loop 0" if self.GifLoopChecker.isChecked() else ""} '
                          f'{Tools.fillQuotation(output_v)} -y').replace("\\", "/")

        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        self.chores_thread = UiRunThread(ffmpeg_command, data={"type": "GIF制作"})
        self.chores_thread.run_signal.connect(self.function_update_chores_finish)
        self.chores_thread.start()
        self.GifButton.setEnabled(False)

    def function_get_SuperResolution_paths(self, path_type=0, key_word=""):
        """
        获取超分路径
        :param key_word: should be module name
        :param path_type: 0: algo 1: module
        :return:
        """
        ncnn_dir = os.path.join(appDir, "ncnn")
        sr_ncnn_dir = os.path.join(ncnn_dir, "sr")
        if path_type == 0:
            return sr_ncnn_dir
        if path_type == 1:
            return os.path.join(sr_ncnn_dir, key_word, "models")

    def function_load_tasks_settings(self, load_all=False, load_one=False):
        task_data = self.InputFileName.getItems()
        task_list = list()
        self.function_disable_inputfilename_connection()

        for t in task_data:
            if self.InputFileName.itemWidget(t).iniCheck.isChecked():
                row_ = self.InputFileName.getWidgetData(t)['row']
                self.InputFileName.setCurrentRow(row_)
                self.on_InputFileName_currentItemChanged()
                task_list.append(row_)
        if load_all or (not len(task_list) and self.InputFileName.count() >= 1):
            """
            Activate All Tasks when load_all is assigned or 
            multiple inputs with no check in (reckoned as mis-operation)
            """
            self.on_InputFileName_currentItemChanged()  # save last modified task
            for it in range(self.InputFileName.count()):
                self.InputFileName.setCurrentRow(it)
                self.on_InputFileName_currentItemChanged()
            task_list = list(range(self.InputFileName.count()))
        elif load_one:
            task_current_item = self.InputFileName.currentItem()
            self.on_InputFileName_currentItemChanged()
            task_list.clear()
            if task_current_item is not None:
                task_list.append(self.InputFileName.getWidgetData(task_current_item)['row'])
            pass

        self.function_enable_inputfilename_connection()
        return task_list

    def function_get_templates(self):
        templates = [self.SettingsTemplateSelector.itemText(i) for i in range(self.SettingsTemplateSelector.count())]
        return templates

    def function_check_read_tutorial(self):
        check_tutorial = os.path.join(appDir, "ReadTutorial.md")
        if not os.path.exists(check_tutorial):
            self.function_send_msg("Read Tutorial First", _translate("", "请先仔细阅读教程再使用软件"))
            self.on_tutorialLinkButton_clicked()
            with open(check_tutorial, "w", encoding="utf-8") as w:
                w.write("# This User Has Read Tutorial")

    def function_disable_inputfilename_connection(self):
        try:
            self.InputFileName.disconnect()
        except:
            pass

    def function_enable_inputfilename_connection(self):
        self.function_disable_inputfilename_connection()
        try:
            self.InputFileName.failSignal.connect(self.on_InputFileName_failImport)
            # self.InputFileName.itemClicked.connect(self.on_InputFileName_currentItemChanged)
            self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
            self.InputFileName.addSignal.connect(self.on_InputFileName_currentItemChanged)
        except:
            print(traceback.format_exc())
            pass

    def steam_update_achv(self):
        if not self.is_steam:
            return
        ACHV_Use_MX250 = self.STEAM.GetAchv("ACHV_Use_MX250")
        ACHV_Use_RTX2060 = self.STEAM.GetAchv("ACHV_Use_RTX2060")
        current_GPU = self.DiscreteCardSelector.currentText()
        if all([i in current_GPU for i in ['MX', '250']]) and not ACHV_Use_MX250:
            reply = self.STEAM.SetAchv("ACHV_Use_MX250")
        if all([i in current_GPU for i in ['RTX', '2060']]) and not ACHV_Use_RTX2060:
            reply = self.STEAM.SetAchv("ACHV_Use_RTX2060")
        self.STEAM.Store()

    def process_update_rife(self, json_data):
        """
        Communicate with RIFE Thread
        :return:
        """

        def generate_error_log():
            self.function_generate_log(0)

        def remove_last_line():
            cursor = self.OptionCheck.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.OptionCheck.setTextCursor(cursor)

        def error_handle():
            now_text = self.OptionCheck.toPlainText().lower() + data.get("subprocess", "").lower()  # 复合寻找错误
            if self.current_failed:
                return
            if "input file is not available" in now_text:
                self.function_send_msg("Inputs Failed", _translate('', "你的输入文件有问题！请检查输入文件是否能够播放，路径有无特殊字符"), )
                self.current_failed = True
                return
            elif "json" in now_text:
                self.function_send_msg("Input File Failed", _translate('', "文件读取失败，请确保软件有足够权限且输入文件未被其他软件占用"), )
                self.current_failed = True
                return
            elif "ascii" in now_text:
                self.function_send_msg("Software Path Failure", _translate('', "请把软件所在文件夹移到纯英文、无中文、无空格路径下"), )
                self.current_failed = True
                return
            elif "cuda out of memory" in now_text:
                self.function_send_msg("CUDA Failed",
                                       _translate('', "你的显存不够啦！去清一下后台占用显存的程序，或者去'高级设置'降低视频分辨率/使用半精度模式/更换补帧模型~"), )
                self.current_failed = True
                return
            elif "cudnn" in now_text.lower() and "fail" in now_text.lower():
                self.function_send_msg("CUDA Failed", _translate('', "请前往官网更新驱动www.nvidia.cn/Download/index.aspx"), )
                self.current_failed = True
                return
            elif "concat test error" in now_text or "concat error" in now_text:
                self.function_send_msg("Concat Failed", _translate('', "区块合并音轨失败，请检查输出文件格式是否支持源文件音频"), )
                self.current_failed = True
                return
            elif "broken pipe" in now_text:
                self.function_send_msg("Render Failed", _translate('', "请检查渲染设置，确保输出分辨率宽高为偶数，尝试关闭硬件编码以解决问题"), )
                self.current_failed = True
                return
            elif "rife_ncnn_vulkan" in now_text:
                self.function_send_msg("NCNN Import Failed", _translate('', "你的A卡不支持NCNN-RIFE补帧，请更换设备"), )
                self.current_failed = True
                return
            elif "Steam Validation Failed" in now_text:
                self.function_send_msg("Steam Validation Failed",
                                       _translate('', "Steam验证失败，请确保软件联网并退出Steam重试；如有疑问详询开发人员"), )
                self.current_failed = True
                return
            elif "error" in now_text:
                logger.error(f"[At the end of One Line Shot]: \n {data.get('subprocess')}")
                __msg1 = _translate('', '程序运行出现错误！')
                __msg2 = _translate('', '请联系开发人员解决')
                self.function_send_msg("Something went wrong",
                                       f"{__msg1}\n{data.get('subprocess')}\n{__msg2}", )
                self.current_failed = True
                return

        data = json.loads(json_data)
        self.progressBar.setMaximum(int(data["cnt"]))
        self.progressBar.setValue(int(data["current"]))
        new_text = ""

        if len(data.get("notice", "")):
            new_text += data["notice"] + "\n"

        if len(data.get("subprocess", "")):
            dup_keys_list = ["Process at", "frame=", "matroska @", "0%|", f"{ArgumentManager.app_id}", "Steam ID",
                             "AppID", "SteamInternal"]
            if 'error' not in data['subprocess'] and any([i in data["subprocess"] for i in dup_keys_list]):
                tmp = ""
                lines = data["subprocess"].splitlines()
                for line in lines:
                    if not any([i in line for i in dup_keys_list]) and len(line.strip()):
                        tmp += line + "\n"
                if tmp.strip() == lines[-1].strip():
                    lines[-1] = ""
                data["subprocess"] = tmp + lines[-1]
                remove_last_line()
            new_text += data["subprocess"]

        for line in new_text.splitlines():
            line = html.escape(line)

            check_line = line.lower()
            if "process at" in check_line:
                add_line = f'<p><span style=" font-weight:600;">{line}</span></p>'
            elif "program finished" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#55aa00;">{line}</span></p>'
            elif "info" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#17C2FF;">{line}</span></p>'
            elif any([i in check_line for i in
                      ["error", "invalid", "incorrect", "critical", "fail", "can't", "can not"]]):
                if all([i not in check_line for i in
                        ["invalid dts", "incorrect timestamps"]]):
                    add_line = f'<p><span style=" font-weight:600; color:#ff0000;">{line}</span></p>'
                else:
                    add_line = f'<p><span>{line}</span></p>'
            elif "warn" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#ffaa00;">{line}</span></p>'
            # elif "duration" in line.lower():
            #     add_line = f'<p><span style=" font-weight:600; color:#550000;">{line}</span></p>'
            else:
                add_line = f'<p><span>{line}</span></p>'
            self.OptionCheck.append(add_line)

        if data["finished"]:
            """Error Handle"""

            if self.chores_thread is not None and len(self.chores_thread.get_main_error()):
                main_error = self.chores_thread.get_main_error()
                self.OptionCheck.append(f"LAST ERROR MSG in OLS:\n{main_error}")

            returncode = data["returncode"]
            complete_msg = f"For {data['cnt']} Tasks:\n"
            if returncode == 0 or "Program Finished" in self.OptionCheck.toPlainText() or (
                    returncode is not None and returncode > 3000):
                """What the fuck is SWIG?"""
                complete_msg += _translate('', '成功！')
                os.startfile(self.OutputFolder.text())
                self.InputFileName.refreshTasks()
            else:
                # if not self.DebugChecker.isChecked():
                _msg1 = _translate('', '失败, 返回码：')
                _msg2 = _translate('', '请将弹出的文件夹内error.txt发送至交流群排疑，并尝试前往高级设置恢复补帧进度')
                complete_msg += f"{_msg1}{returncode}\n{_msg2}"
                error_handle()
                generate_error_log()
            if not self.DebugChecker.isChecked():
                self.function_send_msg(_translate('', "任务完成"), complete_msg, 2)
            self.ConcatAllButton.setEnabled(True)
            self.StartExtractButton.setEnabled(True)
            self.StartRenderButton.setEnabled(True)
            self.AllInOne.setEnabled(True)
            self.InputFileName.setEnabled(True)
            self.current_failed = False

            if appPref.value("use_clear_inputs", False, type=bool):
                self.InputFileName.clear()
            self.function_enable_inputfilename_connection()

        self.OptionCheck.moveCursor(QTextCursor.End)

    # @pyqtSlot(bool)
    def on_InputFileName_currentItemChanged(self):
        current_item = self.InputFileName.currentItem()
        if current_item is None:
            item_count = len(self.InputFileName.getItems())
            if item_count:
                current_item = self.InputFileName.item(0)
            else:
                return
        if self.InputFileName.itemWidget(current_item) is None:  # check if exists this item in InputFileName
            return
        widget_data = self.InputFileName.getWidgetData(current_item)
        if widget_data is not None:
            self.settings_maintain_item_settings(widget_data)  # 保存当前设置，并准备跳转到新任务的历史设置（可能没有）

        self.settings_maintain_io(widget_data)
        return

    def settings_maintain_io(self, widget_data: dict):
        input_path = widget_data.get('input_path')
        if os.path.isfile(input_path):
            input_fps = Tools.get_fps(input_path)
            self.InputFPS.setText(f"{input_fps:.5f}")
            if self.InterpExpReminder.isChecked():  # use exp to calculate outputfps
                try:
                    exp = int(self.ExpSelecter.currentText()[1:])
                    self.OutputFPS.setText(f"{input_fps * exp:.5f}")
                except Exception:
                    pass
        else:
            if not len(self.InputFPS.text()):
                self.InputFPS.setText("0")

    def on_InputFileName_failImport(self, fail_code: int):
        """

        :param fail_code:
        :return:
        """
        if fail_code == 1:
            """Path too long"""
            self.function_send_msg("Path Too Long", _translate('', '输入文件路径过长，请适当缩短文件路径并重试'))
        elif fail_code == 2:
            """Free Version Does not support multi import"""
            self.function_send_msg("Community Version", _translate('', '请升级专业版以使用任务队列'))

    @pyqtSlot(bool)
    def on_InputButton_clicked(self):
        input_files = self.function_select_file(_translate('', '要补帧的视频'), multi=True)
        if not len(input_files):
            return
        for f in input_files:
            self.InputFileName.addFileItem(f)
        if not len(self.OutputFolder.text()):
            self.OutputFolder.setText(os.path.dirname(input_files[0]))

    @pyqtSlot(bool)
    def on_InputDirButton_clicked(self):
        input_directory = self.function_select_file(_translate('', "要补帧的图片序列文件夹"), folder=True)
        self.InputFileName.addFileItem(input_directory)
        if not len(self.OutputFolder.text()):
            self.OutputFolder.setText(os.path.dirname(input_directory))
        return

    @pyqtSlot(bool)
    def on_OutputButton_clicked(self):
        folder = self.function_select_file(_translate('', '要输出项目的文件夹'), folder=True)
        self.OutputFolder.setText(folder)

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        懒人式启动补帧按钮
        :return:
        """
        task_list = self.function_load_tasks_settings()
        if not self.settings_check_args():
            return
        self.settings_load_current()  # update settings

        if appPref.value("is_preview_args", False, type=bool) and not appPref.value("is_gui_quiet", False, type=bool):
            SVFI_preview_args_form = UiPreviewArgsDialog(self)
            SVFI_preview_args_form.setWindowTitle("Preview SVFI Arguments")
            # SVFI_preview_args_form.setWindowModality(Qt.ApplicationModal)
            SVFI_preview_args_form.exec_()
        _msg1 = _translate('', '共有{}个任务，第一个任务将会区块[{}]，起始帧[{}]启动。')
        _msg5 = _translate('', '请确保上述三者皆不为空(-1为自动)，任务计数不为0。\n是否执行补帧？')
        _msg4 = ""
        try:
            current_path = self.InputFileName.itemWidget(self.InputFileName.currentItem()).input_path
            current_ext = os.path.splitext(current_path)[1]
            selected_ext = "." + self.ExtSelector.currentText()
            if current_ext != selected_ext:
                _msg4 = _translate("", "输出文件格式与源不相同，请注意这有可能导致合并失败！")
        except Exception:
            pass
        reply = self.function_send_msg("Confirm Start Info",
                                       _msg1.format(len(task_list), self.StartChunk.text(), self.StartFrame.text()) +
                                       f"\n{_msg5}\n{_msg4}",
                                       3)
        if reply == QMessageBox.No:
            return
        self.AllInOne.setEnabled(False)
        self.InputFileName.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = UiRun(task_list=task_list)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()

        self.rife_thread = RIFE_thread
        _msg1 = _translate('', '补帧操作启动')
        update_text = f"""[SVFI {self.version} {_msg1}]"""
        self.OptionCheck.setText(update_text)
        self.current_failed = False
        self.tabWidget.setCurrentIndex(1)  # redirect to info page
        self.steam_update_achv()

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        """
        自动设置启动信息按钮（点我就完事了）
        :return:
        """
        if self.InputFileName.currentItem() is None or not len(self.OutputFolder.text()):
            self.function_send_msg("Invalid Inputs", _translate('', "请检查你的输入和输出文件夹"))
            return
        self.settings_auto_set()

    @pyqtSlot(bool)
    def on_AddTemplateButton_clicked(self):
        template_name = self.EditTemplateName.text()
        if not len(template_name):
            self.function_send_msg("Invalid Template Name", _translate('', "预设名不能为空~"))
            return
        if template_name in self.function_get_templates():
            self.function_send_msg("Invalid Template Name", _translate('', "预设名不能与已有预设重复~"))
            return
        self.settings_load_config(appDataPath)  # appoint appData to root
        self.settings_load_current()  # update appData to current Settings
        template_config = SVFI_Config_Manager({'input_path': 'Template',
                                               'task_id': f'Template_{template_name}'}, appDir, logger)
        template_config.DuplicateConfig()  # write template settings
        self.SettingsTemplateSelector.addItem(template_name)
        _msg1 = _translate('', '已保存指定预设：')
        self.function_send_msg("New Template Saved", f"{_msg1}{template_name}", 2)

    @pyqtSlot(bool)
    def on_RemoveTemplateButton_clicked(self):
        if not self.SettingsTemplateSelector.count():
            self.function_send_msg("No Templates", _translate('', "预设为空~"))
            return
        template_config = SVFI_Config_Manager({'input_path': 'Template',
                                               'task_id': f'Template_{self.SettingsTemplateSelector.currentText()}'},
                                              appDir, logger)
        self.SettingsTemplateSelector.removeItem(self.SettingsTemplateSelector.currentIndex())
        template_config.RemoveConfig()
        self.function_send_msg("Remove Template", _translate('', "已移除指定预设~"), 2)

    @pyqtSlot(bool)
    def on_UseTemplateButton_clicked(self):
        if not self.SettingsTemplateSelector.count():
            self.function_send_msg("No Templates", _translate('', "预设为空~"))
            return
        template_name = self.SettingsTemplateSelector.currentText()
        if template_name is None:
            self.function_send_msg("Invalid Template", _translate('', "请先指定预设~"))
            return
        template_config = SVFI_Config_Manager({'input_path': 'Template',
                                               'task_id': f'Template_{template_name}'}, appDir, logger)
        config_path = template_config.FetchConfig()
        if config_path is None:
            self.function_send_msg("Invalid Config", _translate('', "指定预设不见啦~"))
            return
        template_config.UpdateRootConfig()
        self.settings_load_config(appDataPath)
        self.settings_initiation(item_update=True, template_update=True)
        self.function_send_msg("Config Loaded", _translate('', "已载入指定预设~"), 2)
        # if not appPref.value("is_gui_quiet", False, type=bool):
        #     SVFI_preview_args_form = UiPreviewArgsDialog(self)
        #     SVFI_preview_args_form.setWindowTitle("Preview SVFI Arguments")
        #     SVFI_preview_args_form.exec_()
        # self.settings_load_config(appDataPath)  # 将appData指针指回root

    @pyqtSlot(bool)
    def on_SettingsPresetsApplyButton_clicked(self):
        """
        载入SVFI官方预设
        :return:
        """
        presets_tuple = self.SettingsPresetsInputs.currentIndex(), self.SettingsPresetsSQ.currentIndex(), self.SettingsPresetsFluency.currentIndex()
        presets_name = f"SVFI_Presets_{''.join(map(lambda x: str(x), presets_tuple))}"
        self.SettingsTemplateSelector.addItem(presets_name)
        self.SettingsTemplateSelector.setCurrentIndex(self.SettingsTemplateSelector.count() - 1)
        self.on_UseTemplateButton_clicked()
        self.SettingsTemplateSelector.removeItem(self.SettingsTemplateSelector.count() - 1)

    @pyqtSlot(bool)
    def on_MBufferChecker_clicked(self):
        """
        使用自定义内存限制
        :return:
        """
        logger.debug("Switch To Manual Assign Buffer Size Mode: %s" % self.MBufferChecker.isChecked())
        self.BufferSizeSelector.setEnabled(self.MBufferChecker.isChecked())

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self):
        if not self.InterpExpReminder.isChecked():
            return
        input_fps = self.InputFPS.text()
        if len(input_fps):
            try:
                self.OutputFPS.setText(f"{float(input_fps) * int(self.ExpSelecter.currentText()[1:]):.5f}")
            except Exception:
                self.function_send_msg(_translate('', "帧率输入有误"),
                                       _translate('', "请确认输入输出帧率为有效数据"))

    @pyqtSlot(bool)
    def on_InterpExpReminder_toggled(self):
        bool_result = not self.InterpExpReminder.isChecked()
        self.OutputFPS.setEnabled(bool_result)
        if not bool_result:
            self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(str)
    def on_SettingsPresetsInputs_currentTextChanged(self):
        if self.SettingsPresetsInputs.currentIndex() == 1:
            self.SettingsPresetsFluency.setEnabled(False)
            self.SettingsPresetsFluency.setCurrentIndex(0)
        else:
            self.SettingsPresetsFluency.setEnabled(True)

    @pyqtSlot(bool)
    def on_UseNCNNButton_clicked(self, clicked=True, silent=False):
        if self.hasNVIDIA and self.UseNCNNButton.isChecked() and not silent:
            reply = self.function_send_msg(_translate('', f"确定使用NCNN？"),
                                           _translate('', f"你有N卡，确定使用A卡/核显？"), 3)
            if reply == QMessageBox.Yes:
                logger.debug("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
            else:
                self.UseNCNNButton.setChecked(False)
        else:
            logger.debug("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
        bool_result = not self.UseNCNNButton.isChecked()
        self.NvidiaRifeSettingsBox.setEnabled(bool_result)
        self.AutoInterpScaleChecker.setEnabled(bool_result)
        self.on_AutoInterpScaleChecker_clicked()

        self.NcnnRifeSettingsBox.setEnabled(not bool_result)
        self.settings_update_rife_model_info()

    @pyqtSlot(str)
    def on_TtaModeSelector_currentTextChanged(self):
        self.TtaIterTimesSelector.setVisible(self.TtaModeSelector.currentIndex() > 0)
        self.settings_free_hide()

    @pyqtSlot(bool)
    def on_UseMultiCardsChecker_clicked(self):
        bool_result = self.UseMultiCardsChecker.isChecked()
        self.DiscreteCardSelector.setEnabled(not bool_result)
        self.SelectedGpuLabel.setText(_translate('', "选择的GPU") if not bool_result else
                                      _translate('', "（使用A卡或核显）拥有的GPU个数"))

    @pyqtSlot(bool)
    def on_UseAiSR_clicked(self):
        use_ai_sr = self.UseAiSR.isChecked()
        self.SrField.setVisible(use_ai_sr)
        if use_ai_sr:
            self.settings_update_sr_algo()
            self.settings_update_sr_model()

    @pyqtSlot(str)
    def on_AiSrSelector_currentTextChanged(self):
        self.settings_update_sr_model()
        bool_result = 'realESR' in self.AiSrSelector.currentText()
        self.TileSizeLabel.setVisible(bool_result)
        self.SrTileSizeSelector.setVisible(bool_result)

    @pyqtSlot(str)
    def on_ResizeTemplate_currentTextChanged(self):
        """
        自定义输出分辨率
        :return:
        """
        current_template = self.ResizeTemplate.currentText()

        width, height = 0, 0
        if "480p" in current_template:
            width, height = 480, 270
        if "720p" in current_template:
            width, height = 1280, 720
        elif "1080p" in current_template:
            width, height = 1920, 1080
        elif "2160p" in current_template:
            width, height = 3840, 2160
        elif "3840p" in current_template:
            width, height = 7680, 4320
        elif "%" in current_template:
            current_item = self.InputFileName.currentItem()
            if current_item is None:
                # self.function_send_msg('Select a item first!', _translate('', '未选中输入项'))
                return
            row = self.InputFileName.getWidgetData(current_item)['row']
            input_files = self.function_get_input_paths()
            sample_file = input_files[row]
            try:
                if not os.path.isfile(sample_file):
                    height, width = 0, 0
                else:
                    input_stream = cv2.VideoCapture(sample_file)
                    width = input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            except Exception:
                height, width = 0, 0
            ratio = int(current_template[:-1]) / 100
            width, height = width * ratio, height * ratio
            self.resize_exp = ratio
        self.ResizeWidthSettings.setValue(width)
        self.ResizeHeightSettings.setValue(height)

    @pyqtSlot(bool)
    def on_AutoInterpScaleChecker_clicked(self):
        """使用动态光流"""
        logger.debug("Switch To Auto Scale Mode: %s" % self.AutoInterpScaleChecker.isChecked())
        bool_result = not self.AutoInterpScaleChecker.isChecked()
        self.InterpScaleSelector.setEnabled(bool_result)
        # self.AutoInterpScaleReminder.setVisible(not bool_result)
        # self.AutoInterpScalePredictSizeSelector.setVisible(not bool_result)

    @pyqtSlot(bool)
    def on_slowmotion_clicked(self):
        self.SlowmotionFPS.setEnabled(self.slowmotion.isChecked())

    @pyqtSlot(bool)
    def on_UseFixedScdet_clicked(self):
        logger.debug("Switch To FixedScdetThreshold Mode: %s" % self.UseFixedScdet.isChecked())

    @pyqtSlot(bool)
    def on_ScedetChecker_clicked(self):
        bool_result = self.ScedetChecker.isChecked()
        self.ScdetSelector.setVisible(bool_result)
        self.UseFixedScdet.setVisible(bool_result)
        self.ScdetMaxDiffSelector.setVisible(bool_result)
        self.ScdetUseMix.setVisible(bool_result)
        self.ScdetOutput.setVisible(bool_result)
        self.on_ExpertMode_changed()
        self.settings_free_hide()

    @pyqtSlot(str)
    def on_DupRmMode_currentTextChanged(self):
        self.DupFramesTSelector.setVisible(
            self.DupRmMode.currentIndex() == 1)  # Single Threshold Duplicated Frames Removal
        self.UseSobelChecker.setVisible(self.DupRmMode.currentIndex() > 1)

    @pyqtSlot(bool)
    def on_ImgOutputChecker_clicked(self):
        """
        Support PNG or TIFF
        :return:
        """
        self.ExtSelector.clear()
        if self.ImgOutputChecker.isChecked():
            self.SaveAudioChecker.setChecked(False)
            self.SaveAudioChecker.setEnabled(False)
            for ext in SupportFormat.img_outputs:
                self.ExtSelector.addItem(ext.strip('.'))
        else:
            self.SaveAudioChecker.setEnabled(True)
            for ext in SupportFormat.vid_outputs:
                self.ExtSelector.addItem(ext.strip('.'))

    @pyqtSlot(bool)
    def on_UseEncodeThread_clicked(self):
        self.EncodeThreadSelector.setVisible(self.UseEncodeThread.isChecked())

    @pyqtSlot(str)
    def on_HwaccelSelector_currentTextChanged(self):
        logger.debug("Switch To HWACCEL Mode: %s" % self.HwaccelSelector.currentText())
        check = self.HwaccelSelector.currentText() == "NVENC"
        self.HwaccelPresetLabel.setVisible(check)
        self.HwaccelPresetSelector.setVisible(check)
        encoders = EncodePresetAssemply.encoder[self.HwaccelSelector.currentText()]
        try:
            self.EncoderSelector.disconnect()
        except Exception:
            pass
        self.EncoderSelector.clear()
        for e in encoders:
            self.EncoderSelector.addItem(e)
        self.EncoderSelector.setCurrentIndex(0)
        self.EncoderSelector.currentTextChanged.connect(self.on_EncoderSelector_currentTextChanged)
        self.on_EncoderSelector_currentTextChanged()

    @pyqtSlot(str)
    def on_EncoderSelector_currentTextChanged(self):
        self.PresetSelector.clear()
        currentHwaccel = self.HwaccelSelector.currentText()
        currentEncoder = self.EncoderSelector.currentText()
        presets = EncodePresetAssemply.encoder[currentHwaccel][currentEncoder]
        for preset in presets:
            self.PresetSelector.addItem(preset)

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tab_index):
        if tab_index in [2, 3]:
            """Step 3"""
            if tab_index == 1:
                self.progressBar.setValue(0)
            logger.info("[Main]: Start Loading Settings")
            self.settings_load_current()

    @pyqtSlot(bool)
    def on_GifButton_clicked(self):
        """
        快速制作GIF
        :return:
        """
        if not self.GifInput.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file(_translate('', '请输入要制作成gif的视频文件'))
            self.GifInput.setText(input_filename)
            self.GifOutput.setText(
                os.path.join(os.path.dirname(input_filename), f"{Tools.get_filename(input_filename)}.gif"))
            return
        self.function_quick_gif()
        pass

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        """
        快速合并音视频
        :return:
        """
        if not self.ConcatInputV.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file(_translate('', '请输入要进行音视频合并的视频文件'))
            self.ConcatInputV.setText(input_filename)
            self.ConcatInputA.setText(input_filename)
            self.OutputConcat.setText(
                os.path.join(os.path.dirname(input_filename), f"{Tools.get_filename(input_filename)}_concat.mp4"))
            return
        self.function_quick_concat()
        pass

    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """
        Only Concat Existed Chunks
        :return:
        """
        task_list = self.function_load_tasks_settings(load_one=True)
        self.settings_load_current()  # update settings
        self.ConcatAllButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = UiRun(concat_only=True, task_list=task_list)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        _msg1 = _translate('', '仅合并操作启动')
        self.OptionCheck.setText(f"[SVFI {self.version} {_msg1}]")

    @pyqtSlot(bool)
    def on_StartExtractButton_clicked(self):
        """
        Only Extract Frames from current input
        :return:
        """
        task_list = self.function_load_tasks_settings(load_one=True)
        self.settings_load_current()
        self.StartExtractButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = UiRun(extract_only=True, task_list=task_list)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        _msg1 = _translate('', '仅拆帧操作启动')
        self.OptionCheck.setText(f"[SVFI {self.version} {_msg1}]")

    @pyqtSlot(bool)
    def on_StartRenderButton_clicked(self):
        """
        Only Render Input based on current settings
        :return:
        """
        task_list = self.function_load_tasks_settings()
        self.settings_load_current()
        self.StartRenderButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = UiRun(render_only=True, task_list=task_list)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        _msg1 = _translate('', '仅渲染操作启动')
        self.OptionCheck.setText(f"[SVFI {self.version} {_msg1}]")

    @pyqtSlot(bool)
    def on_KillProcButton_clicked(self):
        """
        Kill Current Process
        :return:
        """
        if self.rife_thread is not None:
            self.rife_thread.kill_proc_exec()

    @pyqtSlot(bool)
    def on_PauseProcess_clicked(self):
        """
        :return:
        """
        if self.rife_thread is not None:
            self.rife_thread.pause_proc_exec()
            if not self.pause:
                self.pause = True
                self.PauseProcess.setText(_translate('', "继续补帧！"))
            else:
                self.pause = False
                self.PauseProcess.setText(_translate('', "暂停补帧！"))

    @pyqtSlot(bool)
    def on_ShowAdvance_clicked(self):
        bool_result = self.AdvanceSettingsArea.isVisible()

        self.AdvanceSettingsArea.setVisible(not bool_result)
        if not bool_result:
            self.ShowAdvance.setText(_translate('', "隐藏高级设置"))
        else:
            self.ShowAdvance.setText(_translate('', "显示高级设置"))
        self.splitter.moveSplitter(10000000, 1)

    @pyqtSlot(bool)
    def on_SaveCurrentSettings_clicked(self):
        pass

    @pyqtSlot(bool)
    def on_LoadCurrentSettings_clicked(self):
        return

    @pyqtSlot(bool)
    def on_ClearInputButton_clicked(self):
        self.on_actionClearAllVideos_triggered()

    @pyqtSlot(bool)
    def on_OutputSettingsButton_clicked(self):
        self.function_generate_log(1)
        self.function_send_msg("Generate Settings Log Success", _translate('', "设置导出成功！settings.log即为设置快照"), 3)
        pass

    @pyqtSlot(bool)
    def on_RefreshStartInfo_clicked(self):
        self.settings_set_start_info(-1, -1)
        self.StartPoint.setTime(QTime(0, 0, 0))
        self.EndPoint.setTime(QTime(0, 0, 0))
        pass

    @pyqtSlot(bool)
    def on_actionManualGuide_triggered(self):
        SVFI_help_form = UiHelpDialog(self)
        SVFI_help_form.setWindowTitle("SVFI Quick Guide")
        SVFI_help_form.show()

    @pyqtSlot(bool)
    def on_actionAbout_triggered(self):
        SVFI_about_form = UiAboutDialog(self)
        SVFI_about_form.setWindowTitle("About")
        SVFI_about_form.show()

    @pyqtSlot(bool)
    def on_actionPreferences_triggered(self):
        self.SVFI_Preference_form = UiPreferenceDialog()
        self.SVFI_Preference_form.setWindowTitle("Preference")
        self.SVFI_Preference_form.preference_signal.connect(self.on_Preference_changed)
        self.SVFI_Preference_form.show()

    def on_Preference_changed(self):
        self.on_ExpertMode_changed()
        self.settings_load_current()

    def on_ExpertMode_changed(self):
        expert_mode = appPref.value("expert", False, type=bool)
        self.UseFixedScdet.setVisible(expert_mode)
        self.ScdetMaxDiffSelector.setVisible(expert_mode)
        # self.QuickExtractChecker.setVisible(expert_mode)
        self.HDRModeField.setVisible(expert_mode)
        self.RenderSettingsLabel.setVisible(expert_mode)
        self.RenderSettingsGroup.setVisible(expert_mode)
        self.FP16Checker.setVisible(expert_mode)
        self.ReverseChecker.setVisible(expert_mode)
        self.KeepChunksChecker.setVisible(expert_mode)
        # self.AutoInterpScaleChecker.setVisible(expert_mode)
        self.ScdetOutput.setVisible(expert_mode)
        self.ScdetUseMix.setVisible(expert_mode)
        self.DeinterlaceChecker.setVisible(expert_mode)
        self.FastDenoiseChecker.setVisible(expert_mode)
        self.EncodeThreadField.setVisible(expert_mode)

        self.settings_free_hide()
        self.settings_dilapidation_hide()

    @pyqtSlot(bool)
    def on_actionBack2Home_triggered(self):
        self.tabWidget.setCurrentIndex(0)

    @pyqtSlot(bool)
    def on_actionBack2Output_triggered(self):
        self.tabWidget.setCurrentIndex(1)

    @pyqtSlot(bool)
    def on_actionImportVideos_triggered(self):
        self.on_InputButton_clicked()

    @pyqtSlot(bool)
    def on_actionStartProcess_triggered(self):
        if not self.AllInOne.isEnabled():
            self.function_send_msg("Invalid Operation", _translate('', "已有任务在执行"))
            return
        self.on_AllInOne_clicked()

    @pyqtSlot(bool)
    def on_actionStopProcess_triggered(self):
        self.on_KillProcButton_clicked()

    @pyqtSlot(bool)
    def on_actionClearVideo_triggered(self):
        try:
            currentIndex = self.InputFileName.currentRow()
            self.InputFileName.takeItem(currentIndex)
        except Exception:
            self.function_send_msg("Fail to Clear Video", _translate('', "未选中输入项"))

    @pyqtSlot(bool)
    def on_actionQuit_triggered(self):
        sys.exit(0)

    @pyqtSlot(bool)
    def on_actionClearAllVideos_triggered(self):
        self.InputFileName.clear()

    @pyqtSlot(bool)
    def on_actionSaveSettings_triggered(self):
        self.settings_load_current()

    @pyqtSlot(bool)
    def on_actionLoadDefaultSettings_triggered(self):
        appData.clear()
        appPref.clear()
        self.settings_update_pack()
        self.function_send_msg("Load Success", _translate('', "已载入默认设置"), 3)

    @pyqtSlot(bool)
    def on_actionLangZH_triggered(self):
        self.settings_change_lang('zh')

    @pyqtSlot(bool)
    def on_actionLangEN_triggered(self):
        self.settings_change_lang('en')

    @pyqtSlot(bool)
    def on_tutorialLinkButton_clicked(self):
        tutorial_path = os.path.join(appDir, "SVFI_Tutorial.pdf")
        if os.path.exists(tutorial_path):
            try:
                os.startfile(f'"{tutorial_path}"')
            except:
                self.function_send_msg("Unable to open tutorial",
                                       _translate("", "未能打开SVFI教程，请安装pdf阅读器后重试，或在软件根目录下寻找SVFI_tutorial.pdf阅读"))
        else:
            self.function_send_msg("Not Find Tutorial", _translate("", "未能找到SVFI教程"))

    def closeEvent(self, event):
        global appData
        if not self.STEAM.steam_valid:
            event.ignore()
            return
        reply = self.function_send_msg("Quit", _translate('', "是否保存当前设置？"), 3)
        if reply == QMessageBox.Yes:
            self.function_load_tasks_settings(load_all=True)
            self.settings_load_config(appDataPath)
            self.settings_load_current()
            if appPref.value("is_rude_exit", False, type=bool):
                Tools.kill_svfi_related()
                pass
            event.accept()
        else:
            event.ignore()
        pass


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        form = UiBackend()
        form.show()
        app.exec_()
        sys.exit()
    except Exception:
        logger.critical(traceback.format_exc())
        sys.exit()
