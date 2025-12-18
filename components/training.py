import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self,data,cfg):
        self.data = data
        self.cfg = data