import torch
import torch.nn as nn
import auraloss


class ConcordanceCorrelationCoefficientLoss(nn.Module):
    """
    (Lin's) Concordance Correlation Coefficient (CCC) loss
        * for regression problem
    Definition:
        L_c = 1 - rho_c
            = 1 - (2 * sigma_xy^2) / (sigma_x^2 + sigma_y^2 + (miu_x - miu_y)^2)
    Details:
        * L_c: loss function
        * rho_c: concordance correlation coefficient (range: -1 ~ 1)
        * sigma_xy^2: covariance of vector x and vector y
        * sigma_x^2: variance of vector x
        * miu_x: mean value of vector x
    ref:
        https://github.com/end2you/end2you/blob/master/end2you/training/losses.py
    """

    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    @staticmethod
    def _get_moments(data_tensor):
        mean_t = torch.mean(data_tensor)
        var_t = torch.var(data_tensor)
        return mean_t, var_t

    def forward(self, predictions, targets):
        """
        :param predictions:  (torch.Tensor) Predictions of the model
        :param targets: (torch.Tensor) Labels of the data
        :return: loss values
        """
        if self.flatten:
            predictions = predictions.flatten(1)
            targets = targets.flatten(1)

        labels_mean, labels_var = self._get_moments(targets)
        pred_mean, pred_var = self._get_moments(predictions)

        mean_cent_prod = torch.mean((predictions - pred_mean) * (targets - labels_mean))
        rho_c = (2 * mean_cent_prod) / (pred_var + labels_var + torch.square(pred_mean - labels_mean))
        batch_loss = 1 - rho_c
        return batch_loss


class MRSTFTLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(*args, **kwargs)

    def forward(self, x, y):
        loss = self.mrstft(x, y)
        return loss
