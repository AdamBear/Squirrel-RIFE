# -*- coding: utf-8 -*-
import json
import locale
import os
import random
import re
import shutil
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Utils.utils import Tools, ArgumentManager

abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)

class SVFITranslator(QTranslator):
    def __init__(self):
        super().__init__()
        self.app_name = "SVFI"
        try:
            lang = locale.getdefaultlocale()[0].split('_')[0]
            lang_file = self.get_lang_file(lang)
            self.load(lang_file)
        except Exception as e:
            print(e)

    def get_lang_file(self, lang:str):
        lang_file = os.path.join(dname, 'lang', f'SVFI_UI.{lang}.qm')
        return lang_file

    def change_lang(self, lang: str):
        lang_file = self.get_lang_file(lang)
        self.load(lang_file)

class MyListWidgetItem(QWidget):
    dupSignal = pyqtSignal(dict)
    remSignal = pyqtSignal(dict)

    def __init__(self, parent=None):
        """
        Custom ListWidgetItem to display RIFE Task
        :param parent:
        """
        super().__init__(parent)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.filename = QtWidgets.QLabel(self)
        self.iniCheck = QtWidgets.QCheckBox(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename.sizePolicy().hasHeightForWidth())
        # self.iniCheck.setSizePolicy(sizePolicy)
        # self.ini_checkbox.setMinimumSize(QSize(200, 0))
        self.iniCheck.setObjectName("ini_checkbox")
        self.horizontalLayout.addWidget(self.iniCheck)
        self.filename.setSizePolicy(sizePolicy)
        self.filename.setMinimumSize(QSize(400, 0))
        self.filename.setObjectName("filename")
        self.horizontalLayout.addWidget(self.filename)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.line = QtWidgets.QFrame(self)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.RemoveItemButton = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RemoveItemButton.sizePolicy().hasHeightForWidth())
        self.task_id_reminder = QtWidgets.QLabel(self)
        self.task_id_reminder.setSizePolicy(sizePolicy)
        self.task_id_reminder.setObjectName("task_id_reminder")
        self.TaskIdDisplay = MyLineWidget(self)
        self.TaskIdDisplay.setSizePolicy(sizePolicy)
        self.TaskIdDisplay.setObjectName("TaskIdDisplay")
        self.horizontalLayout.addWidget(self.task_id_reminder)
        self.horizontalLayout.addWidget(self.TaskIdDisplay)
        self.RemoveItemButton.setSizePolicy(sizePolicy)
        self.RemoveItemButton.setObjectName("RemoveItemButton")
        self.horizontalLayout.addWidget(self.RemoveItemButton)
        self.DuplicateItemButton = QtWidgets.QPushButton(self)
        self.DuplicateItemButton.setSizePolicy(sizePolicy)
        self.DuplicateItemButton.setObjectName("DuplicateItemButton")
        self.horizontalLayout.addWidget(self.DuplicateItemButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.RemoveItemButton.setText("    -    ")
        self.DuplicateItemButton.setText("    +    ")
        # self.iniCheck.setEnabled(False)
        self.RemoveItemButton.clicked.connect(self.on_RemoveItemButton_clicked)
        self.DuplicateItemButton.clicked.connect(self.on_DuplicateItemButton_clicked)
        self.TaskIdDisplay.editingFinished.connect(self.on_TaskIdDisplay_editingFinished)
        """Item Data Settings"""
        self.task_id = None
        self.input_path = None
        self.check_time = time.time()

    def setTask(self, input_path: str, task_id: str):
        self.task_id = task_id
        self.input_path = input_path
        len_cut = 60
        if len(self.input_path) > len_cut:
            self.filename.setText(self.input_path[:len_cut] + "...")
        else:
            self.filename.setText(self.input_path)
        self.task_id_reminder.setText("  id:")
        self.TaskIdDisplay.setText(f"{self.task_id}")

    def get_task_info(self):
        self.task_id = self.TaskIdDisplay.text()
        return {"task_id": self.task_id, "input_path": self.input_path}

    def on_DuplicateItemButton_clicked(self, e):
        """
        Duplicate Item Button clicked
        action:
            1: duplicate
            0: remove
        :param e:
        :return:
        """
        emit_data = self.get_task_info()
        emit_data.update({"action": 1})
        self.dupSignal.emit(emit_data)
        pass

    def on_RemoveItemButton_clicked(self, e):
        emit_data = self.get_task_info()
        emit_data.update({"action": 0})
        self.dupSignal.emit(emit_data)
        pass

    def on_TaskIdDisplay_editingFinished(self):
        previous_task_id = self.task_id
        emit_data = self.get_task_info()  # update task id by the way
        emit_data.update({"previous_task_id": previous_task_id, "action": 3})
        self.dupSignal.emit(emit_data)  # update

    def on_iniCheck_toggled(self):
        if time.time() - self.check_time < 0.2:
            return
        self.check_time = time.time()
        self.iniCheck.setChecked(not self.iniCheck.isChecked())


class MyLineWidget(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            url = e.mimeData().urls()[0]
            if not len(self.text()):
                self.setText(url.toLocalFile())
        else:
            e.ignore()


class MyListWidget(QListWidget):
    addSignal = pyqtSignal(int)
    failSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.task_dict = list()

    def checkTaskId(self, input_path, task_id):
        """
        Check Task Id Availability
        :param input_path:
        :param task_id:
        :return: bool
        """
        potential_config = SVFI_Config_Manager({'input_path': input_path, "task_id": task_id}, dname)
        if potential_config.FetchConfig() is not None:
            return False
        if task_id in self.task_dict:
            return False
        self.task_dict.append(task_id)
        return True

    def generateTaskId(self, input_path: str):
        path_md5 = Tools.md5(input_path)[:6]
        while True:
            path_id = random.randrange(100000, 999999)
            task_id = f"{path_md5}_{path_id}"
            if self.checkTaskId(input_path, task_id):
                break
        return task_id

    def saveTasks(self):
        """
        return tasks information in strings of json format
        :return: {"inputs": [{"task_id": self.task_id, "input_path": self.input_path}]}
        """
        data = list()
        for item in self.getItems():
            widget = self.itemWidget(item)
            item_data = widget.get_task_info()
            data.append(item_data)
        return json.dumps({"inputs": data})

    def dropEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            for url in e.mimeData().urls():
                item = url.toLocalFile()
                self.addFileItem(item)
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        self.dropEvent(e)

    def getWidgetData(self, item):
        """
        Get widget data from item's widget
        :param item: item
        :return: dict of widget data, including row
        """
        try:
            widget = self.itemWidget(item)
            # widget.on_iniCheck_toggled()
            item_data = widget.get_task_info()
            item_data.update({"row": self.row(item)})
        except AttributeError:
            return None
        return item_data

    def getItems(self):
        """
        获取listwidget中条目数
        :return: list of items
        """
        widgetres = []
        count = self.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.item(i))
        return widgetres

    def refreshTasks(self):
        items = [self.getWidgetData(item) for item in self.getItems()]
        self.clear()
        try:
            for item in items:
                new_item_data = self.addFileItem(item['input_path'])  # do not use previous task id
                config_maintainer = SVFI_Config_Manager(new_item_data, dname)
                config_maintainer.DuplicateConfig(item)  # use previous item data to establish config file

        except RuntimeError:
            pass

    def addConfigItem(self, input_config: str, input_task_id):
        task_id = re.findall("SVFI_Config_(.*?).ini", os.path.basename(input_config))
        if not len(task_id):
            return input_config, input_task_id
        task_id = task_id[0]
        appData = QSettings(input_config, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")
        try:
            input_list_data = json.loads(appData.value("gui_inputs", "{}"))
        except json.decoder.JSONDecodeError:
            return input_config, input_task_id
        for item_data in input_list_data['inputs']:
            if item_data['task_id'] == task_id:
                input_path = item_data['input_path']
                return input_path, task_id
        return None, None

    def addFileItem(self, input_path: str, task_id=None) -> dict:
        input_path = input_path.strip('"')
        if len(input_path) > ArgumentManager.path_len_limit:
            self.failSignal.emit(1)  # path too long
            return {"input_path": input_path, "task_id": task_id}
        if ArgumentManager.is_free:  # in free version, only one task available
            if self.count() >= 1:
                self.failSignal.emit(2)  # community version does not support multi import
                return {"input_path": input_path, "task_id": task_id}
        # input_path, task_id = self.addConfigItem(input_path, task_id)
        if task_id is None:
            task_id = self.generateTaskId(input_path)
        taskListItem = MyListWidgetItem()
        taskListItem.setTask(input_path, task_id)
        taskListItem.dupSignal.connect(self.itemActionResponse)
        taskListItem.remSignal.connect(self.itemActionResponse)
        # Create QListWidgetItem
        taskListWidgetItem = QListWidgetItem(self)
        # Set size hint
        taskListWidgetItem.setSizeHint(taskListItem.sizeHint())
        # Add QListWidgetItem into QListWidget
        self.addItem(taskListWidgetItem)
        self.setItemWidget(taskListWidgetItem, taskListItem)
        item_data = {"input_path": input_path, "task_id": task_id}
        self.addSignal.emit(self.count())
        return item_data

    def itemActionResponse(self, e: dict):
        """
        Respond to item's action(click on buttons)
        :param e:
        :return:
        """
        """
        self.dupSignal.emit({"task_id": self.task_id, "input_path": self.input_path, "action": 1})
        """
        task_id = e.get('task_id')
        target_item = None
        # find target task
        for item in self.getItems():
            task_data = self.itemWidget(item).get_task_info()  # if modify, get task info from new item
            if task_data['task_id'] == task_id:
                target_item = item
                break
        if target_item is None:
            return
        if e.get("action") == 1:  # dupSignal
            input_path = self.itemWidget(target_item).input_path
            new_item_data = self.addFileItem(input_path)
            config_maintainer = SVFI_Config_Manager(new_item_data, dname)  # use new id
            config_maintainer.DuplicateConfig(self.getWidgetData(target_item))
            pass
        elif e.get("action") == 0:  # removeSignal
            self.takeItem(self.row(target_item))
        elif e.get("action") == 3:  # modifySignal
            item_data = self.getWidgetData(target_item)
            config_maintainer = SVFI_Config_Manager(item_data, dname)
            previous_item_data = {"input_path": item_data['input_path'], "task_id": e.get("previous_task_id")}
            config_maintainer.DuplicateConfig(previous_item_data)

    def keyPressEvent(self, e):
        current_item = self.currentItem()
        if current_item is None:
            e.ignore()
            return
        # if e.key() == Qt.Key_Delete:
        #     self.removeItemWidget(current_item)


class MyTextWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                url = event.mimeData().urls()[0]
                self.setText(f"{url.toLocalFile()}")
            else:
                event.ignore()
        except Exception as e:
            print(e)


class MyComboBox(QComboBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MySpinBox(QSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MyDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class SVFI_Config_Manager:
    """
    SVFI 配置文件管理类
    """

    def __init__(self, item_data: dict, app_dir: str, _logger=None):

        self.input_path = item_data['input_path']
        self.task_id = item_data['task_id']
        self.dirname = os.path.join(app_dir, "Configs")
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        self.SVFI_config_path = os.path.join(app_dir, "SVFI.ini")
        self.config_path = self.__generate_config_path()
        if _logger is None:
            self.logger = Tools.get_logger("ConfigManager", "")
        else:
            self.logger = _logger
        pass

    def FetchConfig(self):
        """
        根据输入文件名获得配置文件路径，并将配置文件替换到根配置
        :return:
        """
        if os.path.exists(self.config_path):
            return self.config_path
        else:
            return None

    def DuplicateConfig(self, item_data=None):
        """
        复制配置文件
        将根配置复制到此类对应的位置
        :return:
        """
        if not os.path.exists(self.SVFI_config_path):
            self.logger.warning("Not find Basic Config")
            return False
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if item_data is not None:
            """Duplicate from previous item_data"""
            previous_config_manager = SVFI_Config_Manager(item_data, dname)
            previous_config_path = previous_config_manager.FetchConfig()
            if previous_config_path is not None:
                """Previous Item Data is not None"""
                shutil.copy(previous_config_path, self.config_path)
        else:
            shutil.copy(self.SVFI_config_path, self.config_path)
        return True

    def RemoveConfig(self):
        """
        移除配置文件
        :return:
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        else:
            self.logger.warning("Not find Config to remove, guess executed directly from main file")
        pass

    def UpdateRootConfig(self):
        """
        维护或更新根配置文件,在LoadSettings后维护
        :return:
        """
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, self.SVFI_config_path)
            return True
        else:
            return False
        pass

    def __generate_config_path(self):
        return os.path.join(self.dirname, f"SVFI_Config_{self.task_id}.ini")
