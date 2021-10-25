import argparse
import os
import paddle
import paddle.nn.functional as F
import paddle.vision as vison
import models
from paddle.vision.datasets import Cifar10
import utils

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dataset', type=str, default='Cifar10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/home/aistudio/data', required=False, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', required=False, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--model_path', type=str, default=None, required=True,
                    help='model path (default: None)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
paddle.seed(args.seed)
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(vison.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = Cifar10(mode="test", download=True, transform=model_cfg.transform_test, backend='cv2')
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

criterion = F.cross_entropy
checkpoint = paddle.load(args.model_path)
model.load_dict(checkpoint['state_dict'])
test_res = utils.eval(loaders['test'], model, criterion)
print("model eval accuracy is :", test_res["accuracy"])
print("model eval loss is :", test_res["loss"])






