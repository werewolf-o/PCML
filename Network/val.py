import os
import torch
from torch.autograd import Variable
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, argparse, time, datetime, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util2.MY_dataset import MY_dataset
from util2.augmentation import RandomFlip, RandomCrop
from util2.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util2.lr_policy import WarmUpPolyLR
from util2.init_func import group_weight
from config import config
from model2.LXNet2_xiaorong.相互学习.tu.LXNet_zengqiang import LECSFormer

def inference_and_evaluate(model_path, test_loader, save_dir, device, args, n_class=3):

    current_time = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')

   # save_dir = os.path.join(save_dir, current_time+'_abnormal_test')######
   # save_dir = os.path.join(save_dir, current_time + '_normal_test')  ######
   # save_dir = os.path.join(save_dir, current_time + '_rural_test')  ######
   # save_dir = os.path.join(save_dir, current_time + '_urban_test')  ######
    save_dir = os.path.join(save_dir, current_time + '_test')  ######

    os.makedirs(save_dir, exist_ok=True)


    color_dir = os.path.join(save_dir, 'color')
    comparison_dir = os.path.join(save_dir, 'comparison')
    visual_dir = os.path.join(args.data_dir, 'visual')

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    model = LECSFormer(3,gpu_ids=[0])######
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    conf_total = np.zeros((n_class, n_class))
    color_map = {
        0: [0, 0, 0],  # 未标注 - 黑色
        1: [255, 0, 0],  # 坑洼 - 红色
        2: [0, 255, 0]  # 裂缝 - 绿色
    }
    results_file = os.path.join(save_dir, f'{args.model_name}_inference_log.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(tqdm(test_loader)):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            a= model(images) ################
            predictions = a.argmax(1).cpu().numpy()
            label_batch = labels.cpu().numpy()

            for pred, label in zip(predictions, label_batch):
                conf = confusion_matrix(y_true=label.flatten(),
                                        y_pred=pred.flatten(),
                                        labels=[0, 1, 2])
                conf_total += conf

            for idx, (pred, name) in enumerate(zip(predictions, names)):

                img_name = os.path.splitext(os.path.basename(name))[0]
                visual_name = f"visual{img_name}.png"
                colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for label_id, color in color_map.items():
                    colored_pred[pred == label_id] = color
                color_save_path = os.path.join(color_dir, f'{img_name}.png')
                Image.fromarray(colored_pred).save(color_save_path)

                plt.figure(figsize=(20, 8))
                plt.subplots_adjust(wspace=0.1)

                visual_path = os.path.join(visual_dir, visual_name)
                if os.path.exists(visual_path):
                    visual_img = Image.open(visual_path)
                    plt.subplot(121)
                    plt.imshow(visual_img)
                    plt.title('Original Image', fontsize=14)
                    plt.axis('off')
                else:
                    print(f"Warning: Visual image not found: {visual_path}")
                    plt.subplot(121)
                    plt.title('Original Image (Not Found)', fontsize=14)
                    plt.axis('off')

                plt.subplot(122)
                plt.imshow(colored_pred)
                plt.title('Prediction', fontsize=14)
                plt.axis('off')

                comparison_save_path = os.path.join(comparison_dir, f'{img_name}.png')
                plt.savefig(comparison_save_path,bbox_inches='tight',  dpi=300, pad_inches=0.1)
                plt.close()

    precision, recall, IoU, F1 = compute_results(conf_total)
    mean_precision = np.mean(np.nan_to_num(precision))
    mean_recall = np.mean(np.nan_to_num(recall))
    mean_iou = np.mean(np.nan_to_num(IoU))
    mean_f1 = np.mean(np.nan_to_num(F1))

    with open(results_file, 'w') as f:
        f.write("Model: %s\n" % args.model_name)
        f.write("Mean metrics | m_pre:%0.4f, m_rec:%0.4f, m_IoU:%0.4f, m_F1:%0.4f \n " %
                (100 * mean_precision, 100 * mean_recall, 100 * mean_iou, 100 * mean_f1))

        # 每个类别的详细指标
        for i in range(len(precision)):
            if i == 0:
                f.write('un_pre:%0.4f, un_rec:%0.4f, un_IoU:%0.4f, un_F1:%0.4f \n ' %
                        (100 * precision[i], 100 * recall[i], 100 * IoU[i], 100 * F1[i]))
            if i == 1:
                f.write('pot_pre:%0.4f, pot_rec:%0.4f, pot_IoU:%0.4f, pot_F1:%0.4f \n ' %
                        (100 * precision[i], 100 * recall[i], 100 * IoU[i], 100 * F1[i]))
            if i == 2:
                f.write('cra_pre:%0.4f, cra_rec:%0.4f, cra_IoU:%0.4f, cra_F1:%0.4f\n' %
                        (100 * precision[i], 100 * recall[i], 100 * IoU[i], 100 * F1[i]))

    # 打印结果到控制台
    print('\nInference Results:')
    print('Mean metrics:')
    print(f'Precision: {100 * mean_precision:.4f}%')
    print(f'Recall: {100 * mean_recall:.4f}%')
    print(f'IoU: {100 * mean_iou:.4f}%')
    print(f'F1: {100 * mean_f1:.4f}%')

    print('\nClass-wise results:')
    for i, class_name in enumerate(precision):
        print(f'\n{class_name}:')
        print(f'Precision: {100 * precision[i]:.4f}%')
        print(f'Recall: {100 * recall[i]:.4f}%')
        print(f'IoU: {100 * IoU[i]:.4f}%')
        print(f'F1: {100 * F1[i]:.4f}%')

    print('\nResults saved to:', results_file)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--model_path", type=str, default="/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/model2/LXNet2_xiaorong/相互学习/tu/0.7745289532020387.pth")
    parser.add_argument("--save_dir", type=str, default="/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/model2/LXNet2_xiaorong/相互学习/tu/jieguo")
    parser.add_argument('--data_dir', '-dr', type=str, default='/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/NPO++')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="EISNet")####
    parser.add_argument("--n_class", type=int, default=3)
    args = parser.parse_args()


    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


   # test_dataset = MY_dataset(data_dir=args.data_dir, split='abnormal_test', input_h=288, input_w=512)
    #test_dataset = MY_dataset(data_dir=args.data_dir, split='normal_test', input_h=288, input_w=512)
    #test_dataset = MY_dataset(data_dir=args.data_dir, split='rural_test', input_h=288, input_w=512)
    #test_dataset = MY_dataset(data_dir=args.data_dir, split='urban_test', input_h=288, input_w=512)
    test_dataset = MY_dataset(data_dir=args.data_dir, split='test', input_h=288, input_w=512)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    inference_and_evaluate(
        model_path=args.model_path,
        test_loader=test_loader,
        save_dir=args.save_dir,
        device=device,
        args=args,
        n_class=args.n_class
    )


if __name__ == '__main__':
    main()