import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C

        eps = 1e-9
        input = input.clamp(min=eps, max=1 - eps)
        target = target.type(torch.float).clamp(min=eps, max=1 - eps)

        ce = target * torch.log(input)
        weight = target * (1 - input) ** self.gamma
        loss = -self.alpha * weight * ce
        return loss.mean()


if __name__ == '__main__':
    import numpy as np
    pred = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [.6, .1, .3]])
    pred = torch.from_numpy(pred)

    # label = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # label = np.array([2, 1, 0])
    # label = torch.from_numpy(label)
    # label = torch.unsqueeze(label, dim=1).long()
    label = np.array([[1, 1], [0, 1], [0, 0]])
    label = torch.from_numpy(label)

    focal_loss = FocalLoss(alpha=1., gamma=0.)
    print(focal_loss(pred, torch.zeros(3, 3).scatter_(1, label, 1)))
    import torch.nn as nn

    ce_loss = nn.CrossEntropyLoss()
    # label = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    label = np.array([[1, 1], [0, 1], [0, 0]])
    label = torch.from_numpy(label)
    print(focal_loss(pred, label))