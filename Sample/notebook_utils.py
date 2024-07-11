import json
import os.path
import re
import ipykernel
import requests
import logging
import sys
import tensorflow.keras as keras

from urllib.parse import urljoin

from notebook.notebookapp import list_running_servers


def get_notebook_name():
    """
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.join(ss['notebook_dir'], relative_path)

def get_logger(filename=None):
    if not filename:
        notebook=get_notebook_name()
        base=os.path.basename(notebook)
        filename=os.path.splitext(base)[0]+".log"
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(message)s",
    handlers=[
        logging.FileHandler(filename),
        logging.StreamHandler(sys.stdout)
    ])
    return logging.getLogger()

    class ReportCallback(keras.callbacks.Callback):
        def __init__(self,frequency,use_val=True):
            self.file=log_file
            self.freq=frequency
            self.use_val=use_val
            self.separator=" || "
            if not(self.use_val):
                self.separator="\n"
        def on_epoch_end(self, epoch, logs={}):
            if (epoch % self.freq ==0):
                train_loss=logs["loss"]
                train_acc=logs["acc"]
                print(f"\t{epoch}: TRAIN loss {train_loss:.4f},  acc {train_acc:.4f}",end=self.separator)
                if self.use_val:
                    val_loss=logs["val_loss"]
                    val_acc=logs["val_acc"]
                    print(f"VAL loss {val_loss:.4f}, acc {val_acc:.4f}")


class LoggingCallback(keras.callbacks.Callback):
    def __init__(self,frequency,logger,use_val=True):
        self.logger=logger
        self.freq=frequency
        self.use_val=use_val
        self.separator=" || "
        if not(self.use_val):
            self.separator="\n"
    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.freq ==0):
            train_loss=logs["loss"]
            train_acc=logs["acc"]
            msg=f"\t{epoch}: TRAIN loss {train_loss:.4f},  acc {train_acc:.4f}"
            if self.use_val:
                val_loss=logs["val_loss"]
                val_acc=logs["val_acc"]         
                msg+=f" {self.separator} VAL loss {val_loss:.4f}, acc {val_acc:.4f}"
            self.logger.info(msg)