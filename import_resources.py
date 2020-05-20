import warnings
warnings.filterwarnings('ignore')

import numpy as np
import json
import argparse

from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)