from __future__ import print_function
from math import log10

import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
import torchvision.transforms as transforms

from loadfea import FeaList
from DicNet import ADLnet
# Training settings
parser = argparse.ArgumentParser(description="PyTorch DemoireNet")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument("--nEpochs", type=int, default=100000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=10000,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=250")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--is_debug", action="store_true", help="Use debug path?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients. Default=0.1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--seed", type=int, default=None, help="random seed")

def main():
    global opt, model, training_data_loader, testing_data_loader
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    if opt.is_debug:
        path_train = '/home/wqy/Documents/MvADL-master/nus_train.csv'
        path_test = '/home/wqy/Documents/MvADL-master/nus_test.csv'
        training_data_loader = torch.utils.data.DataLoader(
            FeaList(rootsrc = path_train, roottgt=None,
                      transform=None),
            batch_size=opt.batchSize, shuffle=True)

        testing_data_loader = torch.utils.data.DataLoader(
            FeaList(rootsrc = path_test, roottgt=None,
                      transform=None,is_test = True),
            batch_size=opt.testBatchSize, shuffle=False)

    else:
        path_src = '/home/wqy/Documents/MvADL-master/usps.csv'
        training_data_loader = torch.utils.data.DataLoader(
            FeaList(rootsrc=path_src, roottgt=None,
                    transform=None),
            batch_size=opt.batchSize, shuffle=True)

        testing_data_loader = torch.utils.data.DataLoader(
            FeaList(rootsrc=path_src, roottgt=None,
                    transform=None, is_test=True),
            batch_size=opt.testBatchSize, shuffle=False)

    print("===> Building model")
    model = ADLnet(input_dim1 = 64, input_dim2 = 31)
    model._initialize_weights()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    criterion = nn.NLLLoss()
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(optimizer, criterion, epoch)
        if epoch % 100 == 0:
            save_checkpoint(model, epoch)
            test(epoch)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(optimizer, criterion, epoch):
    epoch_loss = 0
    lr = adjust_learning_rate(epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if epoch % 50 == 0:
        print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()
    nums = 0.0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0], batch[1]
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(input)
        _, predict_id = torch.max(output, 1)
        #_, target_id = torch.max(target, 1)
        target_id =target
        nums += target_id.eq(predict_id.data).cpu().sum().float()
        loss = criterion(output, target)
        epoch_loss += loss.item()

        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        if iteration % 500 == 0:
            loss_record = "===> Epoch[{}]({}/{}): Loss: {:.10f} ".format(epoch, iteration,
                                                                                     len(training_data_loader),
                                                                        loss.item()          )
            with open("train_loss_log.txt", "a") as train_log_file:
                train_log_file.write(loss_record + '\n')
            print(loss_record)
    epoch_loss_record = "===>Training Epoch [{}] Complete: Avg. Entropy Loss: {:.10f}, acc {:.4f}".format(epoch,
                                                                                                          epoch_loss / len(training_data_loader),
                                                                        nums/(len(training_data_loader)*opt.batchSize)
                                                                                                   )
    with open("train_loss_log.txt", "a") as train_log_file:
        train_log_file.write(epoch_loss_record + '\n')
    if epoch % 50 == 0:
        print(epoch_loss_record)


def test(epoch):
    nums = 0.0
    model.eval()
    print("===> Testing")
    with torch.no_grad():
        for iteration, batch in enumerate(testing_data_loader, 1):
            input, target = batch[0], batch[1]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            prediction = model(input,is_test=True)
            _, predict_id = torch.max(prediction, 1)
            #_, target_id = torch.max(target, 1)
            target_id = target
            # print('input', input)
            # print('prediction', prediction)
            # print('predict_id', predict_id)
            # print('target', target_id)
            # use eq func to calculate multiple
            nums = nums + target_id.eq(predict_id.data).cpu().sum().float()

        test_loss_record = "===>Testing Epoch[{}] ACC: {:.4f}".format(epoch,nums / (opt.testBatchSize *len(testing_data_loader)))
        print(test_loss_record)
        with open("test_loss_log.txt", "a") as test_log_file:
            test_log_file.write(test_loss_record + '\n')


def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()