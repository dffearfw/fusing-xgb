import abc
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.process.utils.security import SecureProcessor


class BaseProcessor(metaclass=abc.ABCMeta):
    def __init__(self,config_path,output_dir,secure_processor=None):
        self.config=self.load_config(config_path)
        self.output_dir=Path(output_dir)
        self.secure_processor=secure_processor or SecureProcessor()
        self.logger=_get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True,exist_ok=True)

    def load_config(self,config_path):
        with open(config_path,'r') as f:
            return yaml.safe_load(f)

    @abc.abstractmethod
    def process_date(self,date):
        pass

    def process_range(self,start_date,end_date):
        date_range=pd.date_range(start_date,end_date)
        for date in date_range:
            self.process_date(date)

    def process_all(self):
        start_date=datetime.strptime(self.config['date_range']['start'],'%Y-%m-%d')
        end_date=datetime.strptime(self.config['date_range']['end'],'%Y-%m-%d')
        self.process_range(start_date,end_date)





































