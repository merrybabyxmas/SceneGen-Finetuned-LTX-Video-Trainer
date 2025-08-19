import os
import Pathlib 
from typing import List, Dict, Tuple, Literal
from pydantic import BaseModel
from abc import *
import torch 
import torch.nn as nn 


class BaseSplitter(metaclass = ABCMeta)
