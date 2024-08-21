# -*- coding: utf-8 -*-
# @Time    : 2024/8/13 9:54
# @Author  : Shining
# @File    : utils.py
# @Description :


import logging
import os
from PIL import Image
import sys
import time

LOG_FORMAT = "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(filename=time.strftime("%Y-%d-%M-%H-%M-%S") + ".log", level=logging.DEBUG, format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    '''
    :param exc_type: 错误类型
    :param exc_value: 错误描述
    :param exc_traceback:
    :return:
    '''
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("{}".format(exc_value), exc_info=(exc_type, exc_value, exc_traceback))  # 重点


sys.excepthook = handle_exception  # 重点


def string_is_space_or_empty(s):
    """
    :param s: 路径字符串
    :return: 判断路径字符串是否为空
    """
    if len(s.strip()) == 0:
        return True
    return False



