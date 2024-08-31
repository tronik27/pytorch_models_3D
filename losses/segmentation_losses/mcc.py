import torch
import torch.nn.functional as F
from base_modules import SegmentationLoss




class MCCLoss(SegmentationLoss):
    def __init__(self, eps: float = 1e-5, ignore_value: float = -1):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__(ignore_value=ignore_value)
        self.eps = eps

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the Matthews Correlation Coefficient (MCC) Loss for 3D data.

        This method calculates the MCC loss for binary, multi-binary, and multiclass segmentation tasks
        on 3D volumetric data. The MCC is a balanced measure which can be used even if the classes are 
        of very different sizes.

        Args:
            y_pred (torch.Tensor): The predicted segmentation volume, shape (N, C, D, H, W).
            y_true (torch.Tensor): The ground truth segmentation volume, shape (N, D, H, W) or (N, C, D, H, W).
            filtration_mask (torch.Tensor, optional): A binary mask for filtering regions of interest, shape (N, 1, D, H, W).

        Returns:
            torch.Tensor: The computed MCC loss.

        Note:
            - For binary mode, y_true should be of shape (N, D, H, W) and y_pred of shape (N, 1, D, H, W).
            - For multi-binary mode, both y_true and y_pred should be of shape (N, C, D, H, W).
            - For multiclass mode, y_true should be of shape (N, D, H, W) and y_pred of shape (N, C, D, H, W).
            - The loss is computed as 1 - MCC, so perfect predictions result in a loss of 0.
            - A small epsilon is added to avoid division by zero.
            - This implementation is adapted for 3D volumetric data, typically used in medical imaging tasks.
        """

        bs = y_true.shape[0]

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            
            tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
            tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
            fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
            fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

            numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
            denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

            mcc = torch.div(numerator.sum(), denominator.sum())
            loss = 1.0 - mcc
        
        elif self.mode == "multi-binary":  # multi-binary mode
            losses = list()
            for i in range(y_pred.shape[1]):
                y_pred_i = y_pred[:, i:i+1, ...]
                y_true_i = y_true[:, i:i+1, ...]
                
                tp = torch.sum(torch.mul(y_pred_i, y_true_i)) + self.eps
                tn = torch.sum(torch.mul((1 - y_pred_i), (1 - y_true_i))) + self.eps
                fp = torch.sum(torch.mul(y_pred_i, (1 - y_true_i))) + self.eps
                fn = torch.sum(torch.mul((1 - y_pred_i), y_true_i)) + self.eps

                numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
                denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

                mcc = torch.div(numerator.sum(), denominator.sum())
                losses.append(1.0 - mcc)
            
            loss = torch.mean(torch.stack(losses))
            
        elif self.mode == "multiclass":
            num_classes = y_pred.shape[1]
            y_true = F.one_hot(y_true.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            
            losses = list()
            for i in range(num_classes):
                y_pred_i = y_pred[:, i:i+1, ...]
                y_true_i = y_true[:, i:i+1, ...]
                
                tp = torch.sum(torch.mul(y_pred_i, y_true_i)) + self.eps
                tn = torch.sum(torch.mul((1 - y_pred_i), (1 - y_true_i))) + self.eps
                fp = torch.sum(torch.mul(y_pred_i, (1 - y_true_i))) + self.eps
                fn = torch.sum(torch.mul((1 - y_pred_i), y_true_i)) + self.eps

                numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
                denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

                mcc = torch.div(numerator.sum(), denominator.sum())
                losses.append(1.0 - mcc)
            
            loss = torch.mean(torch.stack(losses))

        return loss
