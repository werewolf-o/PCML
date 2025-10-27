import os, argparse, time, datetime, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
# import torchvision.utils as vutils
# from util.MY_dataset import MY_dataset
from util.My import MY_dataset
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util.lr_policy import WarmUpPolyLR
from util.init_func import group_weight
from config import config
from model.SPNet import SPNet
from model.TCNet import TCNet
from collections import defaultdict
from tr.loss import CrossNetInteraction,MultiscaleLoss
from Train.NPO.loss import PDDM,SuperTeacherDistillation
# from Loss.cwd import ChannelWiseDivergence
from Train.lovasz_losses import lovasz_softmax

parser = argparse.ArgumentParser(description='Train with pytorch')
parser.add_argument('--model_name', '-m', type=str, default='')
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--lr_start', '-ls', type=float, default=6e-5)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=300)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='')
args = parser.parse_args()




# Example usage


metrics = defaultdict(float)
def cc_loss(pre, mask):
    mask = torch.sigmoid(mask)
    pre = torch.sigmoid(pre)
    intersection = (pre * mask).sum(axis=(2, 3))
    unior = (pre + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


class Loss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, temperature=1.0):
        super(Loss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        # self.L1 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.metrics = metrics
        self.iou = iou_loss
        self.interaction = CrossNetInteraction()
        self.pddm = PDDM()
        self.dadloss4 = SuperTeacherDistillation()
        self.dadloss3 = SuperTeacherDistillation()
        self.dadloss2 = SuperTeacherDistillation()
        self.dadloss1 = SuperTeacherDistillation()
        # self.dice = MultiClassDiceLoss()


        self.MultiscaleLoss = MultiscaleLoss()


    def forward(self, pred_A, pred_B, labels):
        x_B,aux1_B,aux2_B,aux3_B,aux4_B,F4_B,F3_B,F2_B,F1_B = pred_B
        x_A,aux1_A,aux2_A,aux3_A,aux4_A,F4_A,F3_A,F2_A,F1_A = pred_A

        Temp = 1  # 蒸馏温度
        # alpha = 0.6
        # p = 0.2

        # hard loss --------------------------------------------------------------
        loss_A = self.cross_entropy(x_A, labels)
        loss_B = self.cross_entropy(x_B, labels)


        loss_hard = loss_B + loss_A
        # hard loss --------------------------------------------------------------

        # midfuture loss --------------------------------------------------------------
        feats_A = [F4_B,F3_B,F2_B,F1_B]
        feats_B = [F4_A,F3_A,F2_A,F1_A]

        lr_feat_A, lr_feat_B, enhanced_A, enhanced_B = self.interaction(feats_A, feats_B)

        midloss = self.MultiscaleLoss(lr_feat_A, lr_feat_B, enhanced_A, enhanced_B, labels)
        # midfuture loss --------------------------------------------------------------

        # decoderfuture loss --------------------------------------------------------------
        MPDD = self.pddm(x_A, x_B)
        loss_aux1 = self.dadloss1(aux4_A, aux4_B, MPDD, labels)
        loss_aux2 = self.dadloss2(aux3_A, aux3_B, MPDD, labels)
        loss_aux3 = self.dadloss3(aux2_A, aux2_B, MPDD, labels)
        loss_aux4 = self.dadloss4(aux1_A, aux1_B, MPDD, labels)
        loss_aux = loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4
        # decoderfuture loss --------------------------------------------------------------

        # soft = Temp * Temp * self.soft_loss(F.softmax(x_A / Temp),F.softmax(x_B / Temp))  # distillation loss KL散度
        total_loss = loss_hard + loss_aux + midloss

        return total_loss


def train(epo, model_A, model_B, train_loader, optimizer_A, optimizer_B):
    model_A.train()
    model_B.train()
    loss_seg_sum = 0
    criterion = Loss().cuda(args.gpu)

    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        start_t = time.time()
        optimizer_A.zero_grad()
        optimizer_B.zero_grad()

        out_A = model_A(images)
        out_B = model_B(images)

        loss = criterion(out_A, out_B, labels)

        loss.backward()
        optimizer_A.step()
        optimizer_B.step()

        loss_seg_sum += loss.item()

        # Learning rate adjustment
        current_idx = (epo - 0) * config.niters_per_epoch + it
        lr = lr_policy.get_lr(current_idx)

        for param_group in optimizer_A.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_B.param_groups:
            param_group['lr'] = lr

        current_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
        print(
            ' %s | Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f' \
            % (current_time, args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr,
               len(names) / (time.time() - start_t), float(loss)))

        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])

        accIter['train'] = accIter['train'] + 1


def validation(epo, model_A, model_B, val_loader):
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            out_A = model_A(images)
            out_B = model_B(images)

            loss_A = F.cross_entropy(out_A[0], labels)
            loss_B = F.cross_entropy(out_B[0], labels)

            current_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            print('%s | Val: %s, epo %s/%s, iter %s/%s, loss_A %.4f, loss_B %.4f' \
                  % (current_time, args.model_name, epo, args.epoch_max, it + 1, len(val_loader),
                     float(loss_A), float(loss_B)))

            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss_A', loss_A, accIter['val'])
                writer.add_scalar('Validation/loss_B', loss_B, accIter['val'])
            accIter['val'] += 1


def testing(epo, model_A, model_B, test_loader, results_dir):
    global best_iou_A
    global best_iou_B
    global best_iou_epoch_A
    global best_iou_epoch_B

    label_list = ["unlabeled", "pothole", "crack"]
    model_A.eval()
    model_B.eval()

    conf_total_A = np.zeros((args.n_class, args.n_class))
    conf_total_B = np.zeros((args.n_class, args.n_class))

    testing_results_file = os.path.join(results_dir, f'{args.model_name}_log.txt')

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            out_A = model_A(images)
            out_B = model_B(images)

            label = labels.cpu().numpy().squeeze().flatten()
            prediction_A = out_A[0].argmax(1).cpu().numpy().squeeze().flatten()
            prediction_B = out_B[0].argmax(1).cpu().numpy().squeeze().flatten()

            conf_A = confusion_matrix(y_true=label, y_pred=prediction_A, labels=[0, 1, 2])
            conf_B = confusion_matrix(y_true=label, y_pred=prediction_B, labels=[0, 1, 2])

            conf_total_A += conf_A
            conf_total_B += conf_B
            current_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            print('%s | Test: %s, epo %s/%s, iter %s/%s' % (
                current_time, args.model_name, epo, args.epoch_max, it + 1, len(test_loader)))

    precision_A, recall_A, IoU_A, F1_A = compute_results(conf_total_A)
    precision_B, recall_B, IoU_B, F1_B = compute_results(conf_total_B)

    mean_iou_A = IoU_A.mean()
    mean_iou_B = IoU_B.mean()

    if mean_iou_A > best_iou_A:
        best_iou_A = mean_iou_A
        best_iou_epoch_A = epo

    # Save best models
    if mean_iou_A > 0.78:
        checkpoint_file_A = os.path.join(results_dir, f'model_A_{epo}_{mean_iou_A}.pth')
        print(f'New best Model A! Saving checkpoint to: {checkpoint_file_A}')
        torch.save(model_A.state_dict(), checkpoint_file_A)

    writer.add_scalar('Test_A/average_recall', recall_A.mean(), epo)
    writer.add_scalar('Test_A/average_IoU', IoU_A.mean(), epo)
    writer.add_scalar('Test_A/average_precision', precision_A.mean(), epo)
    writer.add_scalar('Test_A/average_F1', F1_A.mean(), epo)

    for i in range(len(precision_A)):
        writer.add_scalar("Test_A(class)/precision_class_%s" % label_list[i], precision_A[i], epo)
        writer.add_scalar("Test_A(class)/recall_class_%s" % label_list[i], recall_A[i], epo)
        writer.add_scalar('Test_A(class)/Iou_%s' % label_list[i], IoU_A[i], epo)
        writer.add_scalar('Test_A(class)/F1_%s' % label_list[i], F1_A[i], epo)

    if mean_iou_B > best_iou_B:
        best_iou_B = mean_iou_B
        best_iou_epoch_B = epo

    # For Model B
    if mean_iou_B > 0.78:
        checkpoint_file_B = os.path.join(results_dir, f'model_B_{epo}_{mean_iou_B}.pth')
        print(f'New best Model B! Saving checkpoint to: {checkpoint_file_B}')
        torch.save(model_B.state_dict(), checkpoint_file_B)

    writer.add_scalar('Test_B/average_recall', recall_B.mean(), epo)
    writer.add_scalar('Test_B/average_IoU', IoU_B.mean(), epo)
    writer.add_scalar('Test_B/average_precision', precision_B.mean(), epo)
    writer.add_scalar('Test_B/average_F1', F1_B.mean(), epo)

    for i in range(len(precision_B)):
        writer.add_scalar("Test_B(class)/precision_class_%s" % label_list[i], precision_B[i], epo)
        writer.add_scalar("Test_B(class)/recall_class_%s" % label_list[i], recall_B[i], epo)
        writer.add_scalar('Test_B(class)/Iou_%s' % label_list[i], IoU_B[i], epo)
        writer.add_scalar('Test_B(class)/F1_%s' % label_list[i], F1_B[i], epo)

    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("Conf | model_name： %s, initial lr: %s, batch size: %s \n" % (
                args.model_name, args.lr_start, args.batch_size))

    with open(testing_results_file, 'a') as f:
        current_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')

        # Model AGMAML resul
        f.write(f"\nEpoch {epo} | {current_time} :\n")
        f.write(' Best m_IoU_A:%0.4f (epoch %d) | m_pre:%0.4f, m_rec:%0.4f, m_IoU:%0.4f, m_F1:%0.4f || ' %
                (100 * best_iou_A, best_iou_epoch_A,
                 100 * np.mean(np.nan_to_num(precision_A)), 100 * np.mean(np.nan_to_num(recall_A)),
                 100 * np.mean(np.nan_to_num(IoU_A)), 100 * np.mean(np.nan_to_num(F1_A))))

        for i in range(len(precision_A)):
            if i == 0:
                f.write('un_pre:%0.4f, un_rec:%0.4f, un_IoU:%0.4f, un_F1:%0.4f || ' % (
                    100 * precision_A[i], 100 * recall_A[i], 100 * IoU_A[i], 100 * F1_A[i]))
            if i == 1:
                f.write('pot_pre:%0.4f, pot_rec:%0.4f, pot_IoU:%0.4f, pot_F1:%0.4f || ' % (
                    100 * precision_A[i], 100 * recall_A[i], 100 * IoU_A[i], 100 * F1_A[i]))
            if i == 2:
                f.write('cra_pre:%0.4f, cra_rec:%0.4f, cra_IoU:%0.4f, cra_F1:%0.4f \n' % (
                    100 * precision_A[i], 100 * recall_A[i], 100 * IoU_A[i], 100 * F1_A[i]))

        # Model B results
        f.write(' Best m_IoU_B:%0.4f (epoch %d) | m_pre:%0.4f, m_rec:%0.4f, m_IoU:%0.4f, m_F1:%0.4f || ' %
                (100 * best_iou_B, best_iou_epoch_B,
                 100 * np.mean(np.nan_to_num(precision_B)), 100 * np.mean(np.nan_to_num(recall_B)),
                 100 * np.mean(np.nan_to_num(IoU_B)), 100 * np.mean(np.nan_to_num(F1_B))))

        for i in range(len(precision_B)):
            if i == 0:
                f.write('un_pre:%0.4f, un_rec:%0.4f, un_IoU:%0.4f, un_F1:%0.4f || ' % (
                    100 * precision_B[i], 100 * recall_B[i], 100 * IoU_B[i], 100 * F1_B[i]))
            if i == 1:
                f.write('pot_pre:%0.4f, pot_rec:%0.4f, pot_IoU:%0.4f, pot_F1:%0.4f || ' % (
                    100 * precision_B[i], 100 * recall_B[i], 100 * IoU_B[i], 100 * F1_B[i]))
            if i == 2:
                f.write('cra_pre:%0.4f, cra_rec:%0.4f, cra_IoU:%0.4f, cra_F1:%0.4f \n' % (
                    100 * precision_B[i], 100 * recall_B[i], 100 * IoU_B[i], 100 * F1_B[i]))

    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '\n'), epo)


def setup_experiment_folder():
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_folder_name = f"{current_time} - {args.model_name}"
    results_dir = os.path.join(weight_dir, new_folder_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save both Model files
    shutil.copy("", os.path.join(results_dir, ""))
    shutil.copy("", os.path.join(results_dir, ""))
    shutil.copy("", os.path.join(results_dir, ""))
    return results_dir


if __name__ == '__main__':
    best_iou_A = 0.0
    best_iou_B = 0.0
    best_iou_epoch_A = 0
    best_iou_epoch_B = 0
    torch.cuda.set_device(args.gpu)

    # Initialize both models

    model_A = SPNet()
    model_B = TCNet()

    base_lr = args.lr_start

    if args.gpu >= 0:
        model_A.cuda(args.gpu)
        model_B.cuda(args.gpu)

    # Setup optimizers for both models
    params_list_A = []
    params_list_B = []
    params_list_A = group_weight(params_list_A, model_A, nn.BatchNorm2d, base_lr)
    params_list_B = group_weight(params_list_B, model_B, nn.BatchNorm2d, base_lr)

    optimizer_A = torch.optim.AdamW(params_list_A, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    optimizer_B = torch.optim.AdamW(params_list_B, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    # Setup directories and logging
    weight_dir = os.path.join("")
    writer = SummaryWriter("")

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MY_dataset(data_dir=args.data_dir, split='train', input_h=288,input_w=512)
    val_dataset = MY_dataset(data_dir=args.data_dir, split='validation', input_h=288, input_w=512)
    test_dataset = MY_dataset(data_dir=args.data_dir, split='test', input_h=288, input_w=512)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = (datetime.datetime.now().
                      replace(microsecond=0))
    accIter = {'train': 0, 'val': 0}
    results_dir = setup_experiment_folder()
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model_A, model_B, train_loader, optimizer_A, optimizer_B)
        validation(epo, model_A, model_B, val_loader)
        testing(epo, model_A, model_B, test_loader, results_dir)

