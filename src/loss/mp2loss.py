import torch
from src.loss.losses import FocalLoss, RegL1Loss

class MP2Loss(torch.nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.opt = opt

        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        
        self.hm_weigth = 1
        self.wh_weigth = 0.1
        self.off_weight = 1
        self.track_weight = 1
        self.seq_weight = 1

    def forward(self, outputs, targets):
        hm_loss, wh_loss, off_loss, track_loss, seq_loss = 0, 0, 0, 0, 0

        hm_loss = self.crit(outputs['hm'], targets['hm'])

        wh_loss = self.crit_reg(outputs['wh'], targets['wh'],
                                targets['ind'], targets['reg_mask'])
        
        off_loss = self.crit_reg(outputs['reg'], targets['reg'],
                                targets['ind'], targets['reg_mask'])

        for i in range(self.opt.num_seq - 1):
            track_loss += self.crit_reg(outputs['dis'][:, i], targets['dis_mask'][:, i],
                                       targets['dis_ind'][:, i], targets['dis']) / (self.opt.num_seq - 1)

        for i in range(self.opt.num_seq):
            seq_loss += self.crit_reg(outputs['hm_seq'][:, i], targets['hm_seq'][:, i]) / self.opt.num_seq


        loss = (self.hm_weigth * hm_loss +
                self.wh_weigth * wh_loss +
                self.off_weight * off_loss +
                self.track_weight * track_loss +
                self.seq_weight * seq_loss)
                
        loss_stats = {
            'loss': loss.item(),
            'hm_loss': hm_loss.item(),
            'wh_loss': wh_loss.item(),
            'off_loss': off_loss.item(),
            'track_loss': track_loss.item(),
            'seq_loss': seq_loss.item()
        }
        return loss, loss_stats