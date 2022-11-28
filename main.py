from utils import *
from NetworkFunction import *
import argparse
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_ImageNet
from Models.ResNet import *
from Models.VGG import *
import torch
import random
import os
import numpy as np


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/datasets', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='/home/saved_models/', help='Directory where the model is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=300, help='Training Epochs of ANNs')
    parser.add_argument('--activation_floor', type=str, default='QCFS', help='ANN activation modules')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=50, help='Batch size')
    parser.add_argument('--L', type=int, default=4, help='Quantization level of QCFS')
    parser.add_argument('--sim_len', type=int, default=32, help='Simulation length of SNNs')
    parser.add_argument('--presim_len', type=int, default=4, help='Pre Simulation length of SRP')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/datasets/cluster/public/ImageNet/ILSVRC2012_train', help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/datasets/cluster/public/ImageNet/ILSVRC2012_val', help='Directory where the ImageNet test dataset is saved')    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    
    cls = 100
    cap_dataset = 10000
    
    if args.dataset == 'CIFAR10':
        cls = 10
    elif args.dataset == 'CIFAR100':
        cls = 100
    elif args.dataset == 'ImageNet':
        cls = 1000
        cap_dataset = 50000
    
    
    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls)
    else:
        error('unable to find model ' + args.arch)
    
    model = replace_maxpool2d_by_avgpool2d(model)
    
    if args.activation_floor == 'QCFS':
        model = replace_activation_by_floor(model, args.L)
    else:
        error('unable to find activation floor: ' + args.activation_floor)
    
    if args.dataset == 'CIFAR10':
        train, test = PreProcess_Cifar10(args.datadir, args.batchsize)
    elif args.dataset == 'CIFAR100':
        train, test = PreProcess_Cifar100(args.datadir, args.batchsize)
    elif args.dataset == 'ImageNet':
        train, test = PreProcess_ImageNet(args.datadir, args.batchsize, train_dir=args.train_dir, test_dir=args.test_dir)
    else:
        error('unable to find dataset ' + args.dataset)


    if args.load_model_name != 'None':
        print(f'=== Load Pretrained ANNs ===')
        model.load_state_dict(torch.load(args.load_model_name + '.pth'))  
    if args.direct_training is True:
        print(f'=== Start Training ANNs ===')
        save_name = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_L' + str(args.L) + '.pth'
        model = train_ann(train, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device, save_name=save_name)

    
    print(f'=== ANNs accuracy after the first training stage ===')
    acc = eval_ann(test, model, args.device)
    print(f'Pretrained ANN Accuracy : {acc / cap_dataset}')
    print(f'=== SNNs accuracy after the SRP stage ===')


    replace_activation_by_MPLayer(model,presim_len=args.presim_len,sim_len=args.sim_len)

    
    if args.presim_len > 0:
        new_acc = mp_test(test, model, net_arch=args.net_arch, presim_len=args.presim_len, sim_len=args.sim_len, device=args.device)
    else:         
        replace_MPLayer_by_neuron(model)
        new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device)

    t = 1
    while t < args.sim_len:
        print(f'time step {t}, Accuracy = {(new_acc[t-1] / cap_dataset):.4f}')
        t *= 2
    print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len-1] / cap_dataset):.4f}')
    
 