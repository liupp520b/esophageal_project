import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = ""
        for k, v in results.items():
            if k!="Class IoU" and k!="Class Precision" and k!="Class Recall(Sensitive)" and k!="Class Specificity" and k!="Class F1 Score" and  k!="Class Dice":
                string += "%s: %.5f\n"%(k, v)
            elif k=="Class Precision":
                for a, b in v.items():
                    string += "%s_Precision: %.5f   "%(a,b)
                string += "\n"
            elif k== "Class Recall(Sensitive)":
                for a, b in v.items():
                    string += "%s_Recall(Sensitive): %.5f   "%(a,b)
                string += "\n"
            elif k=="Class Specificity":
                for a, b in v.items():
                    string += "%s_Specificity: %.5f   "%(a,b)
                string += "\n"
            elif k=="Class F1 Score":
                for a, b in v.items():
                    string += "%s_F1_Score: %.5f   "%(a,b)
                string += "\n"
            elif k=="Class Dice":
                for a, b in v.items():
                    string += "%s_Dice: %.5f   " % (a, b)
                string += "\n"
            elif k=="Class IoU":
                for a, b in v.items():
                    string += "Class_%s_IoU: %.5f   "%(a,b)
                # string += "\n"
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        np.seterr(divide='ignore', invalid='ignore')
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()   # 像素精度

        cls_acc= np.diag(hist) / hist.sum(axis=1)  # 类像素精度
        cls_acc = np.nan_to_num(cls_acc)
        Mean_cls_acc = np.mean(cls_acc)           # 均类像素精度

        cls_precision = np.diag(hist) / hist.sum(axis=0)   # 类精确率、类查准率
        cls_precision = np.nan_to_num(cls_precision)
        Mean_precision = np.mean(cls_precision)     # 均精确率、均查准率

        cls_recall = np.diag(hist) / hist.sum(axis=1)   # 类Recall = 类灵敏度
        cls_recall = np.nan_to_num(cls_recall)
        Mean_recall = np.mean(cls_recall)          # Recall = 灵敏度

        cls_specificity = (np.diag(hist).sum()-np.diag(hist)) / \
                          ((np.diag(hist).sum()-np.diag(hist)) + (hist.sum(axis=0)-np.diag(hist)))  # 类特异性
        cls_specificity = np.nan_to_num(cls_specificity)
        Mean_specificity = np.mean(cls_specificity)  # 特异性

        cls_F1_score = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall)     # 类F1-score
        cls_F1_score = np.nan_to_num(cls_F1_score)
        Mean_F1_score = np.mean(cls_F1_score)      # 均F1-score

        cls_Dice = 2 * np.diag(hist) / (hist.sum(axis=0)+ hist.sum(axis=1))  # 类Dice系数
        cls_Dice = np.nan_to_num(cls_Dice)
        Mean_Dice = np.mean(cls_Dice)     # 均Dice系数

        cls_iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        cls_iou = np.nan_to_num(cls_iou)
        Mean_IoU = np.mean(cls_iou)

        freq = hist.sum(axis=1) / hist.sum()
        FWIoU= (freq[freq > 0] * cls_iou[freq > 0]).sum()  # 频权交并比

        cls_precision = dict(zip(range(self.n_classes), cls_precision))   # 类精确率、类查准率
        cls_recall = dict(zip(range(self.n_classes), cls_recall))    # 类Recall = 类灵敏度
        cls_specificity = dict(zip(range(self.n_classes), cls_specificity))   # 类特异性
        cls_F1_score = dict(zip(range(self.n_classes), cls_F1_score))   # 类F1-score
        cls_Dice = dict(zip(range(self.n_classes), cls_Dice))    # 类Dice系数
        cls_iou = dict(zip(range(self.n_classes), cls_iou))  # 类IoU

        return {
                "Pixel Acc": acc,    # 像素精度
                "Mean Class Acc": Mean_cls_acc,  # 均类像素精度
                "Mean Precision(说明: 查准率)": Mean_precision,  # 均精确率、均查准率
                "Class Precision": cls_precision,  # 类精确率、类查准率
                "Mean Recall(Sensitive, 说明: 查全率)": Mean_recall,     # Recall = 灵敏度
                "Class Recall(Sensitive)": cls_recall,     # 类Recall = 类灵敏度
                "Mean Specificity(说明: 误检率)": Mean_specificity,    # 特异性
                "Class Specificity": cls_specificity,    # 类特异性
                "Mean F1 Score": Mean_F1_score,   # 均F1-score
                "Class F1 Score": cls_F1_score,   # 类F1-score
                "Mean Dice": Mean_Dice,    # 均Dice系数
                "Class Dice": cls_Dice,    # 类Dice系数
                "FWIoU": FWIoU,    # 频权交并比
                "Mean IoU": Mean_IoU,  # mIoU
                "Class IoU": cls_iou, # 类IoU
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
