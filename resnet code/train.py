import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
# from google.colab import drive  # include this on Google Colab
import torchvision.models as models
import sys
sys.path.insert(0, 'Ranger-Deep-Learning-Optimizer/ranger')  # chage the path to your ranger optimizer's path
from ranger import Ranger
from resnet_features import resnet18_features
from tqdm import tqdm
import random


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='imagenette', help='which dataset')
parser.add_argument('--adv', default='normal', help='normal/adv training')
parser.add_argument('--filter', default='None', help='which filter to use')
parser.add_argument('--layer', type=int, default=0, help='which layer to apply filter')

args = parser.parse_args()

if args.adv  == 'adv':
    adv_train = True
else:
    adv_train = False


epsilon = 1.0/255
k = 20
alpha = 0.00784

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


batch_size = 64

train_loader = torch.utils.data.DataLoader(datasets.ImageFolder('data/' + args.data + '/train',
                                                                transform=transforms.Compose([
                                                                    transforms.RandomResizedCrop(224),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])
                                                                    
                                                                ])),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('data/' + args.data + '/val',
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Resize([224, 224]),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])
                                                               ])),
                                          batch_size=batch_size, shuffle=True)
# change the imagefolder path to your datasets' path
classes = ('tench', 'springer', 'casette_player', 'chain_saw',
           'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')


model = resnet18_features(pretrained=False, filter=args.filter, filter_layer=args.layer)

print(model)

if adv_train:
    adversary = LinfPGDAttack(model)

# model = nn.DataParallel(model, device_ids=[0])

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Ranger(model.parameters(), lr=8e-03, eps=1e-06)
# for resuming checkpoints:
# checkpoint = torch.load('drive/MyDrive/Imagenette Classification/ckpt_211.pth', map_location='cuda:0')
# model.load_state_dict(checkpoint['net'])
# optimizer.load_state_dict(checkpoint['optimizer'])

log_name = args.data + '_' + args.adv + '_' + args.filter + '_' + str(args.layer)

best_acc = 0

for epoch in range(200):
    model.train()
    for _, (image, label) in enumerate(tqdm(train_loader)):
        image, label = image.cuda(), label.cuda()
        r = random.uniform(0, 1)
        if adv_train and r >= 0.5:
            adv = adversary.perturb(image, label)
        if adv_train and r >= 0.5:
            y = model(adv)
        else: 
            y = model(image)
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    correct = 0
    total = 0
    for _, (image, label) in enumerate(tqdm(test_loader)):
        image, label = image.cuda(), label.cuda()
        y = model(image)
        pred = torch.argmax(y, dim=1)
        correct += torch.sum((label == pred).long()).cpu().numpy()
        total += image.size(0)
    epoch_acc = correct / total
    print('epoch %d: %4f' % (epoch, correct / total))
    with open('logs/'+log_name, 'a') as f:
        f.write('Epoch: {}, Acc: {}\n'.format(epoch, correct/total))
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if epoch_acc >= best_acc:
        best_acc = epoch_acc
        torch.save(checkpoint, 'models/'+log_name+'.pth')
with open('logs/'+log_name, 'a') as f:
    f.write('Best Acc: {}\n'.format(best_acc))