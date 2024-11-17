import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

TEMPERATURE = 4
CE_WEIGHT = 0.1
KD_WEIGHT = 0.9


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher):
        super(KD, self).__init__(student, teacher)
        self.temperature = TEMPERATURE
        self.ce_loss_weight = CE_WEIGHT
        self.kd_loss_weight = KD_WEIGHT

        self.loss = nn.CrossEntropyLoss()

    def forward(self, image, target, feature=False,**kwargs):
        logits_student,_= self.student(image)
        with torch.no_grad():
            logits_teacher,tfeature= self.teacher(image)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        if feature:
            return logits_student,loss_ce,loss_kd,tfeature,logits_teacher
        else:
            return logits_student,loss_ce,loss_kd