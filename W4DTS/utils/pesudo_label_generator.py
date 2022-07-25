import numpy as np
import torch.nn as nn
import torch
import MinkowskiEngine as ME

class PesudoLabelGenerator_v1_8_ot_rev(nn.Module):
    def __init__(self, in_channels, out_channels, kp=1, dimension=3,memory_history = True):
        super(PesudoLabelGenerator_v1_8_ot_rev, self).__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp = kp
        self.memory_history = memory_history

    def alginment(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[1]).cuda(),
                poses[0].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]
    def alginment_re(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[0]).cuda(),
                poses[1].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]

    def cosine_distance(self,x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def pairwise_distances_l2(self,x, y):
        dist = (x.view(-1, 1, 3) - y.view(1, -1, 3)).pow(2).sum(-1)
        return torch.clamp(dist, 0.0, np.inf)

    def sv_group(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], x.shape[-1]])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), x,)
            ) / sv_info[2][0].cuda().view(-1, 1)

    def sv_group_label(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], self.out_channels])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), torch.eye(self.out_channels)[x.flatten().long()].cuda(),)
            ) / sv_info[2][0].cuda().view(-1, 1)


    def ot(self,p1,p2,f1,f2,):


        pass

    def forward(
        self,
        sur_sv_feature,
        sur_sv_coords,
        sur_sv_gt,
        sv_prob,
        mean_features,
        ori_coords,
        posses,
        sv_gt_idx,
        lg=None,
        val= False,
        tmp_flag = True,
        rev= 1,
        iter = 10,
        epsilon = 0.03
    ):
        # print(sur_sv_coords.shape)

        if not rev:
            ori_coords_in = self.alginment_re(ori_coords, posses)
            al_sur_sv_coords = sur_sv_coords
        else:
            ori_coords_in = ori_coords
            al_sur_sv_coords = self.alginment(sur_sv_coords, posses)
        feat_dist_cos = self.cosine_distance(mean_features, sur_sv_feature)
        coords_dist = self.pairwise_distances_l2(ori_coords_in, al_sur_sv_coords)

        coords_dist = coords_dist/(0.5**2)
        support = (coords_dist < 10 ** 2).float()

        # coords_dist = torch.clamp(coords_dist, 0.2)
        onehot = torch.eye(self.out_channels)[sur_sv_gt.long().view(-1)].cuda()
        dist_matric = 1+ feat_dist_cos- torch.exp(-0.5*coords_dist)
        K= torch.exp(-dist_matric/ epsilon).cuda()*support
        # power = gamma / (gamma + 0.03)

        u2 = torch.ones([K.shape[0]]).cuda()/K.shape[0]
        u1 = torch.ones([K.shape[1]]).cuda()/K.shape[1]
        a = torch.ones([K.shape[1]]).cuda()/K.shape[1]
                # b = torch.ones([K.shape[0]]).cuda()/K.shape[1]
        for _ in range(iter):
            b = u2/ (torch.matmul(K, u1) + 1e-16)
            a = u1/ (torch.matmul(K.T, u2) + 1e-16)

        simlarity = torch.matmul(torch.matmul(torch.diag(a),K.T),torch.diag(b))[:lg]#.min(1)[1]K.T#
        # simlarity = feat_dist_cos * torch.clamp(coords_dist, 0.2)#+coords_dist*prob_dist
        update_sv_prob = sv_prob#torch.log(torch.softmax(sv_prob,dim=1))#*0.5**100*100
        sv_prob_updated = torch.softmax(update_sv_prob, dim=1)
        sm_index = simlarity[:lg].max(1)[1]
        tmpx = onehot[:lg].max(1)[1].long()
        tmp =torch.diag(sv_prob_updated[sm_index].index_select(1,tmpx) , 0)>0.1
        if not tmp.sum():
            tmp = torch.diag(sv_prob_updated[sm_index].index_select(1, tmpx), 0) > 0.0
        sm_index = sm_index[tmp]
        sv_prob_updated[sm_index] = onehot[:lg][tmp]
        sm_index = torch.unique(sm_index)
        trust_svm = sm_index
        pre = sv_prob_updated.max(1)[1]
        sur_sv_feature = mean_features#[trust_svm]
        sur_sv_coords = ori_coords#[trust_svm]
        sur_sv_gt = pre#[trust_svm]
        if not val:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt,sv_prob_updated, sm_index
        else:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt, None,sv_prob_updated,sm_index

class PesudoLabelGenerator_v1_8_ot_rev_poss(nn.Module):
    def __init__(self, in_channels, out_channels, kp=1, dimension=3,memory_history = True):
        super(PesudoLabelGenerator_v1_8_ot_rev_poss, self).__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp = kp
        self.memory_history = memory_history

    def alginment(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[1]).cuda(),
                poses[0].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]
    def alginment_re(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[0]).cuda(),
                poses[1].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def sv_group(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], x.shape[-1]])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), x,)
            ) / sv_info[2][0].cuda().view(-1, 1)

    def sv_group_label(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], self.out_channels])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), torch.eye(self.out_channels)[x.flatten().long()].cuda(),)
            ) / sv_info[2][0].cuda().view(-1, 1)


    def ot(self,p1,p2,f1,f2,):


        pass

    def forward(
        self,
        sur_sv_feature,
        sur_sv_coords,
        sur_sv_gt,
        sv_prob,
        mean_features,
        ori_coords,
        posses,
        sv_gt_idx,
        lg=None,
        val= False,
        tmp_flag = True,
        rev= 1
    ):
        # print(sur_sv_coords.shape)

        if not rev:
            ori_coords_in = self.alginment_re(ori_coords, posses)
            al_sur_sv_coords = sur_sv_coords
        else:
            ori_coords_in = ori_coords
            al_sur_sv_coords = self.alginment(sur_sv_coords, posses)
        # al_sur_sv_coords = self.alginment(sur_sv_coords, posses)
        # feat_dist = self.pairwise_distances_l2(mean_features, sur_sv_feature)
        feat_dist_cos = self.cosine_distance(mean_features, sur_sv_feature)
        # feat_dist = feat_dist/(0.7**2)#torch.std(torch.sqrt(feat_dist))
        coords_dist = self.pairwise_distances_l2(ori_coords_in, al_sur_sv_coords)

        coords_dist = coords_dist/(0.5**2)
        support = (coords_dist < 10 ** 2).float()

        # coords_dist = torch.clamp(coords_dist, 0.2)
        onehot = torch.eye(self.out_channels)[sur_sv_gt.long().view(-1)].cuda()
        dist_matric = 1+ feat_dist_cos- torch.exp(-0.5*coords_dist)

        K= torch.exp(-dist_matric/0.03).cuda()*support
        # power = gamma / (gamma + 0.03)

        u2 = torch.ones([K.shape[0]]).cuda()/K.shape[0]
        u1 = torch.ones([K.shape[1]]).cuda()/K.shape[1]
        a = torch.ones([K.shape[1]]).cuda()/K.shape[1]
                # b = torch.ones([K.shape[0]]).cuda()/K.shape[1]
        for _ in range(10):
            b = u2/ (torch.matmul(K, u1) + 1e-16)
            a = u1/ (torch.matmul(K.T, u2) + 1e-16)

        simlarity = torch.matmul(torch.matmul(torch.diag(a),K.T),torch.diag(b))[:lg]
        update_sv_prob = sv_prob

        sv_prob_updated = torch.softmax(update_sv_prob, dim=1)
        sm_index = simlarity[:lg].max(1)[1]
        tmpx = onehot[:lg].max(1)[1].long()
        tmp =torch.diag(sv_prob_updated[sm_index].index_select(1,tmpx) , 0)>0.1

        if not tmp.sum():
            tmp = torch.diag(sv_prob_updated[sm_index].index_select(1, tmpx), 0) > 0.0
        sm_index = sm_index[tmp]
        sv_prob_updated[sm_index] = onehot[:lg][tmp]
        sm_index = torch.unique(sm_index)

        trust_svm = sm_index
        pre = sv_prob_updated.max(1)[1]
        sur_sv_feature = mean_features#[trust_svm]
        sur_sv_coords = ori_coords#[trust_svm]
        sur_sv_gt = pre#[trust_svm]
        if not val:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt,sv_prob_updated, sm_index
        else:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt, None,sv_prob_updated,sm_index

class PesudoLabelGenerator_v1_8_ot_rev_o(nn.Module):
    def __init__(self, in_channels, out_channels, kp=1, dimension=3,memory_history = True):
        super(PesudoLabelGenerator_v1_8_ot_rev_o, self).__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kp = kp
        self.memory_history = memory_history

    def alginment(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[1]).cuda(),
                poses[0].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]
    def alginment_re(self, coords, poses):
        with torch.no_grad():
            diff_one = torch.matmul(
                torch.inverse(poses[0]).cuda(),
                poses[1].cuda(),
            )
            tmp = torch.ones([coords.shape[0], self.dimension + 1]).cuda()
            tmp[:, :3] = coords
            aligned_coords = torch.matmul(diff_one, tmp.T).T
        return aligned_coords[:, :3]

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def sv_group(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], x.shape[-1]])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), x,)
            ) / sv_info[2][0].cuda().view(-1, 1)

    def sv_group_label(self, x, sv_info):
        with torch.no_grad():
            return (
                torch.zeros([sv_info[0][0].shape[0], self.out_channels])
                .cuda()
                .index_add_(0, sv_info[1][0].cuda(), torch.eye(self.out_channels)[x.flatten().long()].cuda(),)
            ) / sv_info[2][0].cuda().view(-1, 1)


    def ot(self,p1,p2,f1,f2,):


        pass

    def forward(
        self,
        sur_sv_feature,
        sur_sv_coords,
        sur_sv_gt,
        sv_prob,
        mean_features,
        ori_coords,
        posses,
        sv_gt_idx,
        lg=None,
        val= False,
        tmp_flag = True,
        rev= 1
    ):
        # print(sur_sv_coords.shape)
        if not rev:
            ori_coords = self.alginment_re(ori_coords, posses)
            al_sur_sv_coords = sur_sv_coords
        else:
            ori_coords = ori_coords
            al_sur_sv_coords = self.alginment(sur_sv_coords, posses)
        feat_dist_cos = self.cosine_distance(mean_features, sur_sv_feature)
        # feat_dist = feat_dist/(0.7**2)#torch.std(torch.sqrt(feat_dist))
        coords_dist = self.pairwise_distances_l2(ori_coords, al_sur_sv_coords)

        coords_dist = coords_dist/(0.5**2)
        # support = (coords_dist < 10 ** 2).float()

        # coords_dist = torch.clamp(coords_dist, 0.2)
        onehot = torch.eye(self.out_channels)[sur_sv_gt.long().view(-1)].cuda()
        dist_matric = 1+ feat_dist_cos- torch.exp(-0.5*coords_dist)

        K= dist_matric.cuda()#*support
        simlarity = K.T[:lg]
        update_sv_prob = sv_prob
        sv_prob_updated = torch.softmax(update_sv_prob, dim=1)
        sm_index = simlarity[:lg].min(1)[1]
        tmpx = onehot[:lg].max(1)[1].long()
        tmp =torch.diag(sv_prob_updated[sm_index].index_select(1,tmpx) , 0)>0.1


        if not tmp.sum():
            tmp = torch.diag(sv_prob_updated[sm_index].index_select(1, tmpx), 0) > 0.0
        sm_index = sm_index[tmp]
        sv_prob_updated[sm_index] = onehot[:lg][tmp]
        sm_index = torch.unique(sm_index)

        trust_svm = sm_index
        pre = sv_prob_updated.max(1)[1]
        sur_sv_feature = mean_features#[trust_svm]
        sur_sv_coords = ori_coords#[trust_svm]
        sur_sv_gt = pre#[trust_svm]
        if not val:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt,sv_prob_updated, sm_index
        else:
            return trust_svm, sur_sv_feature, sur_sv_coords, sur_sv_gt, None,sv_prob_updated,sm_index