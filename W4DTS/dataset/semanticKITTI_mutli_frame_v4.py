
import json

elastic_deformation = False
import MinkowskiEngine as ME

# print(ME.__file__)
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage
from W4DTS.utils.generate_sequential import *
from MinkowskiEngine.utils import sparse_quantize, sparse_collate
from W4DTS.dataset.dataset_mf import BaseDataset

config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "config/semantic-kitti.yaml"))#-part
)
N_CLASSES = len(config['learning_map_inv'])



class SmanticKITTI(BaseDataset):
    mid_data_format = '{}/{}.npy'

    def __init__(self, data_base, batch_size, window_size, scale, mid_label, config = config, type='train',sv_dir=None):

        super(SmanticKITTI, self).__init__(data_base, batch_size, window_size, scale, mid_label,type, config,sv_dir=sv_dir)

    def get_files(self,type):
        input_base = os.path.join(self.data_base, "sequences")
        dirs = glob.glob(os.path.join(input_base, "*"))

        for dir_ in dirs:
            sq = int(dir_[-2:])
            split = [k for k, v in self.config["split"].items() if sq in v][0]
            if not split == self.type:
                continue
            datas = glob.glob(os.path.join(dir_, "velodyne", "*"))
            labels = glob.glob(os.path.join(dir_, "labels", "*"))
            if self.sv_dir:
                sv_masks = glob.glob(os.path.join(self.sv_dir,dir_[-2:], "superV", "*"))
            else:
                sv_masks = glob.glob(os.path.join(dir_, "superV", "*"))
            clib = parse_calibration(os.path.join(dir_, "calib.txt"))
            poses = parse_poses(os.path.join(dir_, "poses.txt"), clib)

            smaple_indexs = glob.glob(os.path.join(self.data_base, "sv_sample_point_v3", dir_[-2:],'*'))
            datas = list(sorted(datas, key=lambda x: int(x[-10:-4])))
            labels = list(sorted(labels, key=lambda x: int(x[-12:-6])))
            sv_masks = list(sorted(sv_masks, key=lambda x: int(x[-10:-4])))
            smaple_indexs = list(sorted(smaple_indexs, key=lambda x: int(x[-11:-5])))

            f = []
            datas = datas
            poses = poses
            for i, data in enumerate(datas):
                pre = i if i + self.window_size < len(datas) else len(datas) - self.window_size
                data_frames = datas[pre: pre + self.window_size]
                data_frames = [data_frames[0]]+ data_frames
                pose_frames = poses[pre: pre + self.window_size]
                pose_frames = [pose_frames[0]]+ pose_frames
                smaple_index = smaple_indexs[pre: pre + self.window_size]
                smaple_index = [smaple_index[0]]+ smaple_index
                label_frames = labels[pre: pre + self.window_size]
                label_frames = [label_frames[0]]+ label_frames
                # sv_frames = sv_masks[pre: pre + self.window_size]
                # sv_frames = [sv_frames[0]]+ sv_frames
                sv_frames = []

                f.append((sq, data_frames, label_frames, smaple_index, pose_frames,sv_frames))
            self.files.extend(f)

    def aug(self,a, seed):
        a = np.matmul(a, seed)
        return a[:, :3]

    def __getitem__(self,idx):
        # pos = idx#self.files[idx]
        locs = []
        feats = []
        labels_pre = []
        labels_all = []
        poses = []
        reverse=[]
        if self.type=='train':
            seed = np.eye(3) + np.random.randn(3, 3) * 0.01
            seed[0][0] *= np.random.randint(0, 2) * 2 - 1
            # seed *= shift_scale #scale
            theta = np.random.rand() * 2 * math.pi
            seed = np.matmul(
                seed,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )
        file_list = self.files[idx]
        for f in range(self.window_size+1):
            poses.append(file_list[4][f])

            scan = np.fromfile(file_list[1][f], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            coords = scan[:, :3]
            r = scan[:, 3] - np.min(scan[:, 3])  # np.ones_like(scan[:, 3] )#
            if self.type == 'train':
                coords = self.aug(coords, seed)
            coords = coords  # - coords.min(0)

            sample_point= json.load(open(file_list[3][f]))
            p_idx = sample_point['0']+[v for k,v in sample_point.items() if k!='0']
            # sv = np.load(open(file_list[5][f],'rb'))

            coords /= self.scale
            coords = coords.astype(np.int32)

            label = np.fromfile(file_list[2][f], dtype=np.uint32)
            sem_label = label & 0xFFFF
            for k, v in self.config["learning_map"].items():
                sem_label[sem_label == k] = v
            labels_all.append(torch.Tensor(sem_label.astype(np.int)).view(-1, 1))
            # sv_mask = np.isin(sv, p_idx)
            #
            # sem_label[(1-sv_mask).astype(np.bool)] = 0
            # for p_i in p_idx:
            #     p_lab = np.unique(sem_label[sv==p_i],return_counts=True)
            #     if len(p_lab[0])==1:
            #         continue
            #     sem_label[sv == p_i] = p_lab[0][np.argmax(p_lab[1])]
            sem_label = sem_label.astype(np.int)
            # coords, r = sparse_quantize(coords, feats=r.reshape(-1,1),ignore_label=0, quantization_size=scale)sem_mask.astype(np.bool)
            ind = sparse_quantize(
                torch.Tensor(coords),
                features=torch.Tensor(r.reshape(-1, 1)),
                # labels=torch.Tensor(sem_label),
                ignore_label=0,
                return_index=True,
                return_inverse=True,
            )
            coords, r = ind[0], ind[1]

            coords, r = sparse_collate(
                [coords], [r]
            )
            locs.append(coords)
            feats.append(r.view(-1, 1))
            labels_pre.append(torch.Tensor(sem_label).view(-1,1))
            reverse.append(ind[3])

        return {
            'idx' : idx,
            "x": [locs, feats],
            "y": labels_pre,
            "y_all": labels_all,
            "reverse_index": reverse,
            "poses": poses,
        }

    def get_data_loader(self,):
        if self.type == 'train':
            return torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                collate_fn=lambda x: self.train_consist(x),
                num_workers=5,
                shuffle=True,
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                num_workers=5,
                shuffle=False,
            )

    def train_consist(self, x):
        tmp = {}
        # tmp['x']
        for idx, d in enumerate(x):
            if not idx:
                tmp["x"] = d["x"]
                tmp["idx"] = [d["idx"]]
                tmp["min_c"] = [min_c.min(0)[0].view(-1, 4) for min_c in d["x"][0]]
                for j in range(self.window_size + 1):
                    tmp["x"][0][j] -= tmp["min_c"][j]
                tmp["y"] = d["y"]
                tmp["y_all"] = d["y_all"]
                tmp["poses"] = d["poses"]
                tmp["poses"] = d["poses"]
                tmp["reverse_index"] =[ d["reverse_index"]]
                # tmp["poses"][0] = np.eye(4)
            else:
                tmp["idx"].append(d["idx"])

                for j in range(self.window_size + 1):
                    tmp_x = d["x"][0][j]
                    tmp_x[:, 0] += idx
                    tmp["min_c"][j] = torch.cat(
                        (tmp["min_c"][j], d["x"][0][j].min(0)[0].view(-1, 4)),
                    ).view(-1, 4)
                    tmp_x[:, 1:] -= d["x"][0][j].min(0)[0][1:]
                    tmp["x"][0][j] = torch.cat((tmp["x"][0][j], tmp_x), 0)
                    tmp["x"][1][j] = torch.cat((tmp["x"][1][j], d["x"][1][j]), 0)
                    # if j:
                    tmp["y"][j] = torch.cat((tmp["y"][j], d["y"][j]), 0)
                    tmp["y_all"][j] = torch.cat((tmp["y_all"][j], d["y_all"][j]), 0)
                    tmp["poses"][j] = np.append(tmp["poses"][j], d["poses"][j]).reshape(
                            -1, 4, 4
                        )

                tmp["reverse_index"] .append(d["reverse_index"])

        tmp["offset"] = [
            tmp["min_c"][si] - tmp["min_c"][si - 1] if si else tmp["min_c"][si]
            for si in range(len(tmp["min_c"]))
        ]
        return tmp


