import os
import shutil

import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn


class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss")
        self.train_losses = []
        self.val_loss = []
        self.miou = [0]
        self.best_miou = []
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)

    def append_train_loss(self, train_loss):
        self.train_losses.append(train_loss)
        with open(os.path.join(self.save_path, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        self.loss_plot()

    def append_val_loss(self, val_loss):
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def append_miou(self, miou):
        self.miou.append(miou*100)
        with open(os.path.join(self.save_path, "epoch_val_miou.txt"), 'a') as f:
            f.write(str(miou))
            f.write("\n")
        self.miou_plot()

    def miou_plot(self):
        val_iters = [0]
        for i in range(len(self.val_loss)):
            val_iters.append((i+1) * 100)

        plt.plot(val_iters, self.miou, 'g-', linewidth=2, label='miou')
        try:
            if len(self.val_loss) < 25:
                val_num = 5
            else:
                val_num = 15
            plt.plot(val_iters, scipy.signal.savgol_filter(self.miou, val_num, 3), 'b', linestyle='--', linewidth=2,
                     label='smooth miou')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Iters')
        plt.ylabel('miou(%)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_path, "epoch_miou.png"))
        plt.cla()
        plt.close("all")

    def loss_plot(self):
        train_iters = []
        val_iters = []
        for i in range(len(self.train_losses)):
           train_iters.append(i * 10)
        for i in range(len(self.val_loss)):
            val_iters.append((i+1) * 100)
        plt.figure()
        plt.plot(train_iters, self.train_losses, 'red', linewidth=2, label='train loss')
        plt.plot(val_iters, self.val_loss, 'b', linewidth=2, label='val loss')
        try:
            if len(self.train_losses) < 25:
                train_num = 5
            else:
                train_num = 15
            if len(self.val_loss) < 25:
                val_num = 5
            else:
                val_num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_losses, train_num, 3), 'm', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_loss, val_num, 3), 'g', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")


        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
        # plt.show()
        plt.cla()
        plt.close("all")