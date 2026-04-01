# =============================================================================
# utils/logger.py — 日志记录模块
# 同时输出到控制台 (彩色) 和日志文件 (纯文本),方便训练时实时查看与事后回溯
# =============================================================================

import logging
import os
import sys


def get_logger(name: str = "train", log_file: str = None) -> logging.Logger:
    """
    创建并返回一个 Logger 实例。

    参数:
        name     (str): Logger 名称,用于区分不同模块的日志
        log_file (str): 日志文件路径。如果提供,则同时将日志写入文件

    返回:
        logging.Logger: 配置好的 Logger
    """

    logger = logging.getLogger(name)

    # 防止重复添加 handler (多次调用 get_logger 时)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 日志格式: 时间 | 级别 | 消息内容
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 控制台 Handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── 文件 Handler (可选) ───────────────────────────────────────────────────
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
