import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper

# GLFE Module
class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign

        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        # --- multiple local conv3d--- #
        self.local_conv3d_0 = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        self.local_conv3d_1 = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        self.local_conv3d_2 = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        # Global Feature
        global_feat = self.global_conv3d(x)

        # No partation
        if self.halving == 0:
            local_feat = self.local_conv3d(x)
            # Partation A and Partation B
        else:
            # Heigth of Feature Map
            h = x.size(3)

            alpha_1 = h//4
            alpha_2 = h//8
            alpha_3 = h//16
            alpha_4 = h//32

            # partition A
            local_a0 = x[:, :, :, :alpha_1, :]
            local_a1 = x[:, :, :, alpha_1:(alpha_1+alpha_2+alpha_3), :]
            local_a2 = x[:, :, :, (alpha_1 + alpha_2 + alpha_3):(alpha_1 + alpha_2 + 2*alpha_3 + alpha_4), :]
            local_a3 = x[:, :, :, (alpha_1 + alpha_2 + 2 * alpha_3 + alpha_4):(alpha_1 + alpha_2 + 3 * alpha_3 + 2*alpha_4), :]
            local_a4 = x[:, :, :, (alpha_1 + alpha_2 + 3 * alpha_3 + 2 * alpha_4):(alpha_1 + 2*alpha_2 + 4 * alpha_3 + 2 * alpha_4), :]
            local_a5 = x[:, :, :, (alpha_1 + 2 * alpha_2 + 4 * alpha_3 + 2 * alpha_4):(alpha_1 + 2 * alpha_2 + 5 * alpha_3 + 3 * alpha_4), :]
            local_a6 = x[:, :, :, (alpha_1 + 2 * alpha_2 + 5 * alpha_3 + 3 * alpha_4):(alpha_1 + 2 * alpha_2 + 6 * alpha_3 + 4 * alpha_4), :]

            local_feat_a0 = self.local_conv3d_0(local_a0)
            local_feat_a1 = self.local_conv3d_1(local_a1)
            local_feat_a2 = self.local_conv3d_2(local_a2)
            local_feat_a3 = self.local_conv3d_2(local_a3)
            local_feat_a4 = self.local_conv3d_1(local_a4)
            local_feat_a5 = self.local_conv3d_2(local_a5)
            local_feat_a6 = self.local_conv3d_2(local_a6)

            local_feat_a = torch.cat([local_feat_a0, local_feat_a1, local_feat_a2,
                                  local_feat_a3, local_feat_a4, local_feat_a5, local_feat_a6], 3)

            # partition B
            local_b0 = x[:, :, :, :alpha_1, :]
            local_b1 = x[:, :, :, alpha_1:(alpha_1 + alpha_3 + alpha_4), :]
            local_b2 = x[:, :, :, (alpha_1 + alpha_3 + alpha_4):(alpha_1 + 2 * alpha_3 + 2 * alpha_4), :]
            local_b3 = x[:, :, :, (alpha_1 + 2 * alpha_3 + 2 * alpha_4):(alpha_1 + alpha_2 + 3 * alpha_3 + 2 * alpha_4), :]
            local_b4 = x[:, :, :, (alpha_1 + alpha_2 + 3 * alpha_3 + 2 * alpha_4): (alpha_1 + alpha_2 + 4 * alpha_3 + 3 * alpha_4), :]
            local_b5 = x[:, :, :, (alpha_1 + alpha_2 + 4 * alpha_3 + 3 * alpha_4): (alpha_1 + alpha_2 + 5 * alpha_3 + 4 * alpha_4), :]
            local_b6 = x[:, :, :, (alpha_1 + alpha_2 + 5 * alpha_3 + 4 * alpha_4): (alpha_1 + 2 * alpha_2 + 6 * alpha_3 + 4 * alpha_4), :]

            local_feat_b0 = self.local_conv3d_0(local_b0)
            local_feat_b1 = self.local_conv3d_2(local_b1)
            local_feat_b2 = self.local_conv3d_2(local_b2)
            local_feat_b3 = self.local_conv3d_1(local_b3)
            local_feat_b4 = self.local_conv3d_2(local_b4)
            local_feat_b5 = self.local_conv3d_2(local_b5)
            local_feat_b6 = self.local_conv3d_1(local_b6)

            local_feat_b = torch.cat([local_feat_b0, local_feat_b1, local_feat_b2,
                                  local_feat_b3, local_feat_b4, local_feat_b5, local_feat_b6], 3)

            # Local Feature Fusion: partition A + partition B
            local_feat = local_feat_a + local_feat_b

        # GLConvA
        if not self.fm_sign:
            feat = F.leaky_relu(global_feat) + F.leaky_relu(local_feat)
        # GLConvB
        else:
            feat = F.leaky_relu(torch.cat([global_feat, local_feat], dim=3))

        return feat


# GeM
class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)

# GaitSCM
class GaitSCM_OUMVLP(BaseModel):
    def __init__(self, *args, **kargs):
        super(GaitSCM_OUMVLP, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):

        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        
        # === Feature Disentanglement Module === #
        # ID Branch
        self.DIT_ID_0 = SeparateFCs(64, 256, 256)
        self.DIT_Bn_ID_0 = nn.BatchNorm1d(256)
        self.DIT_ID_1 = SeparateFCs(64, 256, 256)
        self.DIT_Bn_ID_1 = nn.BatchNorm1d(256)
        # Non-ID Branch
        self.DIT_NonID_0 = SeparateFCs(64, 256, 256)
        self.DIT_Bn_NonID_0 = nn.BatchNorm1d(256)
        self.DIT_NonID_1 = SeparateFCs(64, 256, 256)
        self.DIT_Bn_NonID_1 = nn.BatchNorm1d(256)

        # ID classifier and Non-ID classifier
        self.Classifier_ID = SeparateFCs(64, 256, 5153)
        self.Classifier_NonID = SeparateFCs(64, 256, 14)
        # Causal Intervention: Backdoor Adjustment
        self.FC_Sum_0 = SeparateFCs(64, 256, 256)
        self.BN_Sum_0 = nn.BatchNorm1d(256)
        self.FC_Sum_1 = SeparateFCs(64, 256, 256)
        self.BN_Sum_1 = nn.BatchNorm1d(256)

        # === Multi-stage training strategy === #
        # === Freeze the parameters of feature disentanglement module and backdoor adjustment === #
        for p in self.parameters():
            p.requires_grad = False

        # === Stage 2 === #
        # === Unfreeze Feature Disentanglement Module === #
        # '''
        # === ID Branch === #
        for p in self.DIT_ID_0.parameters():
            p.requires_grad = True

        for p in self.DIT_Bn_ID_0.parameters():
            p.requires_grad = True

        for p in self.DIT_ID_1.parameters():
            p.requires_grad = True

        for p in self.DIT_Bn_ID_1.parameters():
            p.requires_grad = True

        # === Non-ID Branch === #
        for p in self.DIT_NonID_0.parameters():
            p.requires_grad = True

        for p in self.DIT_Bn_NonID_0.parameters():
            p.requires_grad = True

        for p in self.DIT_NonID_1.parameters():
            p.requires_grad = True

        for p in self.DIT_Bn_NonID_1.parameters():
            p.requires_grad = True

        # === Classifier ID / Non-ID === #
        for p in self.Classifier_ID.parameters():
            p.requires_grad = True

        for p in self.Classifier_NonID.parameters():
            p.requires_grad = True
        # '''

        # === Stage 3 === #
        # === Unfreeze Backdoor Adjustment Module=== #
        # '''
        for p in self.FC_Sum_0.parameters():
            p.requires_grad = True

        for p in self.BN_Sum_0.parameters():
            p.requires_grad = True

        for p in self.FC_Sum_1.parameters():
            p.requires_grad = True

        for p in self.BN_Sum_1.parameters():
            p.requires_grad = True

        # '''


        # === Backbone === #
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )

        # LTA
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 1, 1),
                        stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        # GLFE
        # GLConvA0
        self.GLConvA0 = nn.Sequential(
            GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
            GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
        )

        # Spatial Max Pooling
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # GLConvA1
        self.GLConvA1 = nn.Sequential(
            GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
            GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
        )

        # GLConvB
        self.GLConvB2 = nn.Sequential(
            GLConv(in_c[2], in_c[3], halving=1, fm_sign=False, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
            GLConv(in_c[3], in_c[3], halving=1, fm_sign=True, kernel_size=(3, 3, 3),
                   stride=(1, 1, 1), padding=(1, 1, 1)),
        )

        # Temporal Pooling
        self.TP = PackSequenceWrapper(torch.max)

        # GeM Pooling
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])
        self.Bn = nn.BatchNorm1d(in_c[-1])
        self.Head1 = SeparateFCs(64, in_c[-1], class_num)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        ipts, labs, types, views, seqL = inputs
        device = labs.device

        # === Generate NonID LabelOUMVLP) [14 views = 14] === #
        nonid_type_all = ['000', '015', '030', '045', '060', '075', '090',
                          '180', '195', '210', '225', '240', '255', '270']
        nonid_type_samples = []
        nonid_labels = []
        # types of all samples
        for i in range(len(labs)):
            nonid_type_samples.append(views[i])
        # types of all samples --> labels of all samples
        for i in range(len(nonid_type_samples)):
            for j in range(len(nonid_type_all)):
                if nonid_type_samples[i] == nonid_type_all[j]:
                    nonid_labels.append(j)
        nonid_labels = torch.LongTensor(nonid_labels).to(device)

        # === setting the length of input sequence === #
        seqL = None if not self.training else seqL

        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)


        # ====== Stage1: Feature Extraction Module ===== #
        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)
        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)
        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]
        outs = self.HPP(outs)
        outs = outs.permute(2, 0, 1).contiguous()

        gait = self.Head0(outs)
        gait = gait.permute(1, 2, 0).contiguous()
        bnft = self.Bn(gait)
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())
        gait = gait.permute(0, 2, 1).contiguous()
        bnft = bnft.permute(0, 2, 1).contiguous()
        logi = logi.permute(1, 0, 2).contiguous()

        # === Stage 2: Feature Disentanglement Module === #
        # '''
        dit_outs = bnft.permute(1, 0, 2).contiguous()
        # === ID_branch === #
        # FC0 + BN0 + ReLU
        gait_id = self.DIT_ID_0(dit_outs)
        gait_id = gait_id.permute(1, 2, 0).contiguous()
        bnft_id = self.DIT_Bn_ID_0(gait_id)
        bnft_id = bnft_id.permute(0, 2, 1).contiguous()
        bnft_id = self.relu(bnft_id)
        # FC1 + BN1 + ReLU
        bnft_id = bnft_id.permute(1, 0, 2).contiguous()
        gait_id = self.DIT_ID_1(bnft_id)
        gait_id = gait_id.permute(1, 2, 0).contiguous()
        bnft_id = self.DIT_Bn_ID_1(gait_id)
        bnft_id = bnft_id.permute(0, 2, 1).contiguous()
        bnft_id = self.relu(bnft_id)
        # === NonID_branch === #
        # FC0 + BN0 + ReLU
        gait_nonid = self.DIT_NonID_0(dit_outs)
        gait_nonid = gait_nonid.permute(1, 2, 0).contiguous()
        bnft_nonid = self.DIT_Bn_NonID_0(gait_nonid)
        bnft_nonid = bnft_nonid.permute(0, 2, 1).contiguous()
        bnft_nonid = self.relu(bnft_nonid)
        # FC1 + BN1 + ReLU
        bnft_nonid = bnft_nonid.permute(1, 0, 2).contiguous()
        gait_nonid = self.DIT_NonID_1(bnft_nonid)
        gait_nonid = gait_nonid.permute(1, 2, 0).contiguous()
        bnft_nonid = self.DIT_Bn_NonID_1(gait_nonid)
        bnft_nonid = bnft_nonid.permute(0, 2, 1).contiguous()
        bnft_nonid = self.relu(bnft_nonid)
        # === Classifiers === #
        # == F_ID -- C_ID
        logi_id = bnft_id.permute(1, 0, 2).contiguous()
        logi_id = self.Classifier_ID(logi_id)
        logi_id = logi_id.permute(1, 0, 2).contiguous()
        # == F_Domain -- C_Domain
        logi_nonid = bnft_nonid.permute(1, 0, 2).contiguous()
        logi_nonid = self.Classifier_NonID(logi_nonid)
        logi_nonid = logi_nonid.permute(1, 0, 2).contiguous()
        # === Classifiers [Distinguish] === #
        # == F_ID -- C_Domain
        logi_conf_id = bnft_id.permute(1, 0, 2).contiguous()
        logi_conf_id = self.Classifier_NonID(logi_conf_id)
        logi_conf_id = logi_conf_id.permute(1, 0, 2).contiguous()
        # == F_Domain -- C_ID
        logi_conf_nonid = bnft_nonid.permute(1, 0, 2).contiguous()
        logi_conf_nonid = self.Classifier_ID(logi_conf_nonid)
        logi_conf_nonid = logi_conf_nonid.permute(1, 0, 2).contiguous()
        # '''

        # ======== Stage 3: Backdoor Adjustment ========= #
        # '''
        # Generate R NonID Feature for Each ID Feature
        R = 1  # The number of confounders
        device = bnft_nonid.device
        fea_nonid_n, fea_nonid_p, fea_nonid_c = bnft_nonid.size()
        conf_nonid_all = []

        for i in range(R):
            conf_nonid = torch.ones(1, fea_nonid_p, fea_nonid_c).to(device)
            for j in range(bnft_id.shape[0]):
                N_ = bnft_nonid.shape[0]  # N_ NonID Feature of Each Batch
                N_list = torch.linspace(start=0, end=N_ - 1, steps=N_).int().tolist()
                # Random Sample Two NonID Features from All NonID Features
                N_sample = random.sample(N_list, 2)
                # Mixup Two Random Sampled NonID Feature
                alpha = 0.5
                conf_nonid = torch.cat([conf_nonid,
                                         alpha * bnft_nonid[N_sample[0]].unsqueeze(0) + (1 - alpha) * bnft_nonid[
                                             N_sample[0]].unsqueeze(0)], dim=0)
            # R NonID Mixup Feature u_r
            conf_nonid = conf_nonid[1:, :, :]
            conf_nonid_all.append(conf_nonid)

        Sum_Fea = []
        for i in range(R):
            # Final Feature Fusion [x_i + u_r]
            Sum_Fea.append(bnft_id + conf_nonid_all[i])

        Sum_Fea_n, Sum_Fea_p, Sum_Fea_c = Sum_Fea[0].size()
        class_results_sum_fea_all = torch.ones([1, Sum_Fea_n, Sum_Fea_p, 5153]).to(device)
        fea_ba = Sum_Fea

        for i in range(R):
            Sum_Fea[i] = Sum_Fea[i]
            # Fusion Feature --> Classification ID

            Sum_Fea[i] = Sum_Fea[i].permute(1, 0, 2).contiguous()
            # FC0 + BN0 + ReLU
            Sum_Fea[i] = self.FC_Sum_0(Sum_Fea[i])
            Sum_Fea[i] = Sum_Fea[i].permute(1, 2, 0).contiguous()
            Sum_Fea[i] = self.BN_Sum_0(Sum_Fea[i])
            Sum_Fea[i] = Sum_Fea[i].permute(0, 2, 1).contiguous()
            Sum_Fea[i] = self.relu(Sum_Fea[i])
            # FC1 + BN1 + ReLU
            Sum_Fea[i] = Sum_Fea[i].permute(1, 0, 2).contiguous()
            Sum_Fea[i] = self.FC_Sum_1(Sum_Fea[i])
            Sum_Fea[i] = Sum_Fea[i].permute(1, 2, 0).contiguous()
            Sum_Fea[i] = self.BN_Sum_1(Sum_Fea[i])
            Sum_Fea[i] = Sum_Fea[i].permute(0, 2, 1).contiguous() 
            Sum_Fea[i] = self.relu(Sum_Fea[i])
            # 2 64 256  n p c --> 2 256 64 n c p n c
            Sum_Fea[i] = Sum_Fea[i].permute(1, 0, 2).contiguous()
            class_results_sum_fea = self.Classifier_ID(Sum_Fea[i])
            class_results_sum_fea = class_results_sum_fea.permute(1, 0, 2).contiguous()

            class_results_sum_fea = class_results_sum_fea.unsqueeze(0)
            class_results_sum_fea_all = torch.cat([class_results_sum_fea_all, class_results_sum_fea], dim=0)

        # Final Classification Results
        logi_ba = class_results_sum_fea_all[1:, :, :, :]

        # '''

        n, _, s, h, w = sils.size()

        # Ablation Experiment
        # Stage 1：pretrain feature extraction module
        # === Baseline === #
        '''
        retval = {
            'training_feat': {
                'base_triplet': {'embeddings': bnft, 'labels': labs},
                'base_softmax': {'logits': logi, 'labels': labs}
            },

            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': bnft
            }
        }
        '''


        # Ablation Experiment
        # Stage 2：feature disentanglement module
        # === Disentangle === #
        '''
        retval = {
            'training_feat': {

                'disentangle_triplet_id': {'embeddings': bnft_id, 'labels': labs},
                'disentangle_triplet_nonid': {'embeddings': bnft_nonid, 'labels': nonid_labels},

                'disentangle_softmax_id': {'logits': logi_id, 'labels': labs},
                'disentangle_softmax_nonid': {'logits': logi_nonid, 'labels': nonid_labels},

                'disentangle_conf_id_oumvlp': {'logits': logi_conf_id, 'labels': labs},
                'disentangle_conf_nonid_oumvlp': {'logits': logi_conf_nonid, 'labels': nonid_labels},

            },

            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': bnft_id
            }
        }
        '''

        # Ablation Experiment
        # Stage 3：backdoor adjustment module
        # === Casual Inference [Train] === #
        # '''
        retval = {
            'training_feat': {

                'disentangle_triplet_id': {'embeddings': bnft_id, 'labels': labs},
                'disentangle_triplet_nonid': {'embeddings': bnft_nonid, 'labels': nonid_labels},

                'disentangle_softmax_id': {'logits': logi_id, 'labels': labs},
                'disentangle_softmax_nonid': {'logits': logi_nonid, 'labels': nonid_labels},

                'disentangle_conf_id_oumvlp': {'logits': logi_conf_id, 'labels': labs},
                'disentangle_conf_nonid_oumvlp': {'logits': logi_conf_nonid, 'labels': nonid_labels},

                'casual_softmax_id': {'logits': logi_ba, 'labels': labs},
                'casual_triplet_id': {'embeddings': fea_ba, 'labels': labs},

            },

            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': bnft_id
            }
        }
        # '''

        # Ablation Experiment
        # Stage 3：backdoor adjustment module
        # === Casual Inference [Test] === #
        '''
        retval = {
            'training_feat': {

                'disentangle_triplet_id': {'embeddings': bnft_id, 'labels': labs},
                'disentangle_triplet_nonid': {'embeddings': bnft_nonid, 'labels': nonid_labels},

                'disentangle_softmax_id': {'logits': logi_id, 'labels': labs},
                'disentangle_softmax_nonid': {'logits': logi_nonid, 'labels': nonid_labels},

                'disentangle_conf_id_oumvlp': {'logits': logi_conf_id, 'labels': labs},
                'disentangle_conf_nonid_oumvlp': {'logits': logi_conf_nonid, 'labels': nonid_labels},

            },

            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': bnft_id
            }
        }
        '''
        return retval
