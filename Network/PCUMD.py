import torch
import torch.nn as nn
# from mmkd.Prototype import PrototypeSegmentation
# from semseg.models.segformer.seg_block_UMDt import Seg as Seg_s
# from semseg.models.segformer.seg_block_select_UMD import Seg


def PCUMD(x_all: list, x_all_t: list, lbl: torch.Tensor, prototype_all: list):
    loss_pumd = 0.0
    loss_mse = nn.MSELoss()
    loss_kl = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    B = len(x_all)
    for i in range(B):
        batch_label = lbl[i].unsqueeze(0)

        for j in range(3):

            for k in range(x_all[i][j]):

                x_all_feature = x_all[i][j][k, :, :, :].unsqueeze(0)
                x_all_t_feature = x_all_t[i][j][k, :, :, :].unsqueeze(0)

                x_all_prototype = prototype_all[j].calculate_batch_prototypes(x_all_feature, batch_label)
                x_all_t_prototype = prototype_all[j].calculate_batch_prototypes(x_all_t_feature, batch_label)



                x_all_prototype_log_softmax = torch.log_softmax(x_all_prototype,dim=0)
                x_all_t_prototype_softmax = torch.softmax(x_all_t_prototype, dim=0)
                loss_pumd += loss_kl(x_all_prototype_log_softmax, x_all_t_prototype_softmax).clamp(min=0)



    return loss_pumd / B


# model = Seg("mit_b0", num_classes=25, pretrained=True)
# model_s = Seg_s("mit_b0", num_classes=25, pretrained=True)
#
# sample = [torch.zeros(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024),
#           torch.ones(2, 3, 1024, 1024)]
# lbl = torch.zeros(2, 1024, 1024)
#
# logits, index, ms_feat = model(sample)
# with torch.no_grad():
#     logits_s, ms_feat_s = model_s(sample)
# loss = PUMD(index, ms_feat, ms_feat_s, lbl, model.num_classes)
# print(0)

if __name__ == "__main__":
    from tr.Prototype import PrototypeSegmentation

    embed_dim = [64, 128, 256, 512]
    prototype_all = [PrototypeSegmentation(num_classes=3, feature_dim=embed_dim[0])]
    prototype_all.append(PrototypeSegmentation(num_classes=3, feature_dim=embed_dim[1]))
    prototype_all.append(PrototypeSegmentation(num_classes=3, feature_dim=embed_dim[2]))
    prototype_all.append(PrototypeSegmentation(num_classes=3, feature_dim=embed_dim[3]))
    # 使用您提供的特征尺寸
    feats_A = [torch.randn(2, 64, 72, 128), torch.randn(2, 128, 36, 64),
               torch.randn(2, 256, 18, 32)]
    feats_B = [torch.randn(2, 64, 72, 128), torch.randn(2, 128, 36, 64),
               torch.randn(2, 256, 18, 32)]

    label = torch.randint(0, 3, (3, 288, 512)).cuda(0)  # 随机生成标签

    loss1 = PCUMD(feats_A,feats_B,label)
    print(loss1.item())