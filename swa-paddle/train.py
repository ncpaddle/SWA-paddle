import argparse
import os
import sys
import time
import paddle
import paddle.nn.functional as F
import paddle.vision as vison
import numpy as np
import models
from reprod_log import ReprodLogger
from paddle.vision.datasets import Cifar10
import utils
import tabulate
# from logger import Logger
import logger


parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default='out', required=False, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='Cifar10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='dataset', required=False, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', required=False, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=350, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.05, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.01, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
paddle.set_device("gpu")
def save_model(args):
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    print(model)
    paddle.save(model.state_dict(), 'swa_paddle.pdparams')

def show_pkl(args):
    path_pytorch = "./swa_pytorch.pt"
    path_paddle = "./swa_paddle.pdparams"
    paddle_dict = paddle.load(path_paddle)
    for key in paddle_dict:
        print(key)

def forward_paddle(args):
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    model.load_dict(paddle.load("./model_paddle.pdparams"))
    model.eval()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)
    print(out)
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.save("../diff/forward_paddle.npy")


def metric_paddle(args):
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    model.load_dict(paddle.load("./model_paddle.pdparams"))
    model.eval()

    train_set = Cifar10(mode='train', download=True, transform=model_cfg.transform_train, backend='cv2')
    test_set = Cifar10(mode="test", download=True, transform=model_cfg.transform_test, backend='cv2')
    loaders = {
        'train': paddle.io.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        'test': paddle.io.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    }
    criterion = F.cross_entropy
    result = utils.eval(loaders['test'], model, criterion)
    accuracy = result['accuracy']
    print("accuracy:", accuracy)

    reprod_logger.add("accuracy", np.array(accuracy))
    reprod_logger.save("../diff/metric_paddle.npy")

def loss_paddle(args):
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    model.load_dict(paddle.load("./model_paddle.pdparams"))
    model.eval()
    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    fake_label = np.load("../fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    out = model(fake_data)
    loss = F.cross_entropy(out, fake_label)
    print("loss, ", loss)

    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("../diff/loss_paddle.npy")




def bp_align_paddle(args):
    paddle.set_device("gpu")
    np.random.seed(0)
    paddle.seed(0)
    reprod_logger = ReprodLogger()
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    model.load_dict(paddle.load("./model_paddle.pdparams"))

    # read or gen fake data
    fake_data = np.load("../fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    fake_label = np.load("../fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)
    criterion = F.cross_entropy
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=model.parameters(),
        weight_decay=1e-4,
        momentum=0.9
    )
    model.train()
    loss_list = []
    for idx in range(3):
        fake_data = paddle.to_tensor(fake_data, stop_gradient=True)
        fake_label = paddle.to_tensor(fake_label, stop_gradient=True)
        output = model(fake_data)
        loss = criterion(output, fake_label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss.cpu().detach().numpy())
    print(loss_list)
    print(np.array(loss_list).shape)
    reprod_logger.add("loss_list", np.array(loss_list))
    reprod_logger.save("../diff/bp_align_paddle.npy")

############输出模型############
# save_model(args)
# show_pkl(args)
# forward_paddle(args)
# metric_paddle(args)
# loss_paddle(args)
# p_align_paddle(args)
###############################

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

paddle.seed(args.seed)
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(vison.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
# train_set = ds(path, download=True, transform=model_cfg.transform_train)
train_set = Cifar10(mode='train', download=True, transform=model_cfg.transform_train, backend='cv2')
test_set = Cifar10(mode="test", download=True, transform=model_cfg.transform_test, backend='cv2')
loaders = {
    'train': paddle.io.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ),
    'test': paddle.io.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
}
num_classes = 10

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)


if args.swa:
    print('SWA training')
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_n = 0
else:
    print('SGD training')


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

criterion = F.cross_entropy
optimizer = paddle.optimizer.Momentum(
    learning_rate=args.lr_init,
    parameters=model.parameters(),
    weight_decay=args.wd,
    momentum=args.momentum,
)


start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = paddle.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_dict(checkpoint['state_dict'])
    optimizer.set_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swa_model.state_dict() if args.swa else None,
    swa_n=swa_n if args.swa else None,
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    # utils.adjust_learning_rate(optimizer, lr)
    optimizer.set_lr(lr)

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_res = utils.eval(loaders['test'], swa_model, criterion)
        else:
            swa_res = {'loss': None, 'accuracy': None}
    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            optimizer=optimizer.state_dict()
        )
    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
    if args.swa:
        values = values[:-1] + [swa_res['loss'], swa_res['accuracy']] + values[-1:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]

    with open('log/train_log.txt', 'a') as f:
        f.write(table + '\n')
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )
