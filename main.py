import os, sys
from pathlib import Path

import tensorflow as tf
import threading # 멀티 스레드 사용을 위한 모듈
import json

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

from utils.preprocess import Preprocess
from utils.bot_server import BotServer
from model.intent.intent_model import IntenModel
from model.ner.ner_model import NerModel