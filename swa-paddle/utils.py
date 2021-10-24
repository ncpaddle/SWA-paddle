import os
import paddle


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.ParamAttr:
        param_group['learning_rate'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pdparams' % epoch)
    paddle.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        # input = input.cuda()
        # target = target.cuda()
        # input_var=paddle.create_parameter(input, dtype='float32')
        # target_var = paddle.create_parameter(target, dtype='float32')
        input_var = paddle.to_tensor(input, dtype='float32')
        target_var = paddle.to_tensor(target, dtype='int64')
        output = model(input_var)

        loss = criterion(output, target_var)
        loss.backward()
        
        optimizer.step()
        optimizer.clear_grad()
        loss_sum += loss.numpy().item() * input.shape[0]
        pred = paddle.argmax(output, axis=1, keepdim=True)
        correct += paddle.equal(pred, paddle.reshape(target, pred.shape)).astype('int64').sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input
        target = target
        input_var = paddle.to_tensor(input, dtype='float32')
        target_var = paddle.to_tensor(target, dtype='int64')

        output = model(input_var)
        loss = criterion(output, target_var)

        loss_sum += loss.numpy().item()  * input.shape[0]
        pred = paddle.argmax(output, axis=1, keepdim=True)
        correct += paddle.equal(pred, paddle.reshape(target, pred.shape)).astype('int64').sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def moving_average(net1, net2, alpha=1):
    net1_map = dict(net1.named_parameters())
    for name, parm in net2.named_parameters():
        net1_parm = net1_map[name]
        net1_parm.set_value((1.0-alpha) * net1_parm + alpha * parm)



def _check_bn(module, flag):
    if issubclass(module.__class__, paddle.nn.layer.BatchNorm2D):
        
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__,  paddle.nn.layer.BatchNorm2D):
        module.running_mean = paddle.zeros_like(module.running_mean)
        module.running_var = paddle.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, paddle.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, paddle.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(num_blocking=True)
        input_var = paddle.create_parameter(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))