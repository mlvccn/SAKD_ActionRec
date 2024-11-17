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

class CrossKD(Distiller):
    """CrossKD: Cross-Head Knowledge Distillation for Object Detection"""

    def __init__(self, student, teacher):
        super(CrossKD, self).__init__(student, teacher)
        self.temperature = TEMPERATURE
        self.ce_loss_weight = CE_WEIGHT
        self.kd_loss_weight = KD_WEIGHT
        self.loss = nn.CrossEntropyLoss()

    def align_scale(self, stu_feat, tea_feat):
        N, C = stu_feat.size()  
        _, _ = tea_feat.size() 

        stu_mean = stu_feat.mean(dim=0, keepdim=True)  
        stu_std = stu_feat.std(dim=0, keepdim=True)  
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6) 


        tea_mean = tea_feat.mean(dim=0, keepdim=True)  
        tea_std = tea_feat.std(dim=0, keepdim=True)  

        aligned_feat = stu_feat * tea_std + tea_mean

        return aligned_feat

    
    def forward(self, image, target, feature=False, **kwargs):
        logits_student, student_feature,sf = self.student(image, True)
        with torch.no_grad():
            logits_teacher,teacher_feature,tf = self.teacher(image,True)
            align_feature = self.align_scale(sf,tf)
            logits_studentkd, _ = self.teacher.module.head(align_feature)
        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_studentkd, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        
        if feature:
            return logits_student, loss_ce, loss_kd, teacher_feature, logits_teacher
        else:
            return logits_student, loss_ce, loss_kd
