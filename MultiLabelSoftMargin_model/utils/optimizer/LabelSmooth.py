# Learner: 王振强
# Learn Time: 2022/2/15 17:44
import torch.nn as nn
import torch

'''
    criterion = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
    loss = criterion(out, lbs)
'''


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
        convert labels to one hot vectors
        args:
            labels: torch tensor [label1, label2...]
            classes: int, num of classes
            value: label value in one hot value, defalt to 1
        return:
            return one hot format labels in shape[bs, classes]
        """
        one_hot = torch.zeros(labels.size()[0], classes)
        labels = labels.view(labels.size()[0], -1)
        value_added = torch.Tensor(labels.size()[0], 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """
        args:
            targets: formate [label1, label2, label_batch size]
            length: length of one-hot format (num of classes)
            smooth facter: smooth factor
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length
        return one_hot.to(target.device)




