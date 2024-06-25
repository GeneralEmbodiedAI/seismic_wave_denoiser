import torch.optim as optim

from utils.get_yaml import const


def get_optimizer(models):
    optimizer_type = const['optimizer']['optimizer_type']
    learning_rate = float(const['optimizer']['learning_rate'])
    weight_decay = float(const['optimizer']['weight_decay'])
    momentum = float(const['optimizer']['momentum'])
    params = [{'params': model.parameters()} for model in models]
    if optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    elif optimizer_type == 'sgd':
        # ref: https://zhuanlan.zhihu.com/p/32230623
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum,
                              nesterov=True)
    elif optimizer_type == 'asgd':
        optimizer = optim.ASGD(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('ERROR: optimizer type unknown!')
    return optimizer
