
import json
import random


elastic_deformation = False
import MinkowskiEngine as ME

# print(ME.__file__)
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage
from sklearn.neighbors import KDTree
from W4DTS.utils.generate_sequential import *
from MinkowskiEngine.utils import sparse_quantize, sparse_collate
from W4DTS.dataset.dataset_mf import BaseDataset
import functools

config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "config/semantic-kitti.yaml"))  # -part
)
N_CLASSES = len(config["learning_map_inv"])


class SmanticKITTI(BaseDataset):
    mid_data_format = "{}/{}.npy"

    def __init__(
        self,
        data_base,
        batch_size,
        window_size,
        scale,
        mid_label,
        config=config,
        type="train",
        sv_dir=None,
        sample_interval=None,
        sample_dir="sv_sample_point_v2",
        kp = 1
    ):
        if sample_interval:
            self.sample_interval = sample_interval
        else:
            self.sample_interval = window_size
        self.sample_dir = sample_dir
        super(SmanticKITTI, self).__init__(
            data_base,
            batch_size,
            window_size,
            scale,
            mid_label,
            type,
            config,
            sv_dir=sv_dir,
        )
        self.kp = kp
        self.current_data = []

    def __len__(self):
        return len(self.current_data[0][1])

    def get_files(self, type):
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
                sv_masks = glob.glob(
                    os.path.join(self.sv_dir, dir_[-2:], "superV", "*")
                )
            else:
                sv_masks = glob.glob(os.path.join(dir_, "superV", "*"))
            clib = parse_calibration(os.path.join(dir_, "calib.txt"))
            poses = parse_poses(os.path.join(dir_, "poses.txt"), clib)

            smaple_indexs = glob.glob(
                os.path.join(self.data_base, self.sample_dir, dir_[-2:], "*")
            )
            datas = list(sorted(datas, key=lambda x: int(x[-10:-4])))
            labels = list(sorted(labels, key=lambda x: int(x[-12:-6])))
            sv_masks = list(sorted(sv_masks, key=lambda x: int(x[-10:-4])))
            smaple_indexs = list(sorted(smaple_indexs, key=lambda x: int(x[-11:-5])))

            f = []
            datas = datas
            poses = poses
            for i, data in enumerate(datas):
                if i % self.sample_interval and not self.type == "valid":
                    continue

                pre = (
                    i
                    if i + self.window_size < len(datas)
                    else len(datas) - self.window_size
                )
                length = (self.window_size//2 +1 ) if i + self.window_size <= len(datas) else self.window_size
                data_frames = datas[pre : pre + length]
                # data_frames = [data_frames[0]] + data_frames
                pose_frames = poses[pre : pre + length]
                # pose_frames = [pose_frames[0]] + pose_frames
                smaple_index = smaple_indexs[pre : pre + length]
                # smaple_index = [smaple_index[0]] + smaple_index
                label_frames = labels[pre : pre + length]
                # label_frames = [label_frames[0]] + label_frames
                sv_frames = sv_masks[pre : pre + length]
                # sv_frames = [sv_frames[0]] + sv_frames
                # mid_tmp = os.path.join(
                #     self.mid_label, self.mid_data_format.format(
                #         data_frames[-2].split('/')[-3],
                #         data_frames[-2].split('.')[0].split('/')[-1]),
                # )
                # if not os.path.isfile(mid_tmp):
                    # continue
                f.append(
                        (
                            sq,
                            data_frames,
                            label_frames,
                            smaple_index,
                            pose_frames,
                            sv_frames,
                            1,
                        )
                    )
                if i and (pre % self.sample_interval>=  (self.window_size//2 +1 ) or pre % self.sample_interval==0):
                    data_frames = list(reversed(datas[pre- length+1: pre +1]))
                    # data_frames = [data_frames[0]] + data_frames
                    pose_frames = list(reversed(poses[pre- length+1: pre +1]))
                    # pose_frames = [pose_frames[0]] + pose_frames
                    smaple_index = list(reversed(smaple_indexs[pre- length+1: pre +1]))
                    # smaple_index = [smaple_index[0]] + smaple_index
                    label_frames = list(reversed(labels[pre- length+1: pre +1]))
                    # label_frames = [label_frames[0]] + label_frames
                    sv_frames = list(reversed(sv_masks[pre- length+1: pre +1]))
                    # sv_frames = [sv_frames[0]] + sv_frames

                    mid_tmp = os.path.join(
                        self.mid_label, self.mid_data_format.format(
                            data_frames[-2].split('/')[-3],
                            data_frames[-2].split('.')[0].split('/')[-1]),
                    )
                    # if not os.path.isfile(mid_tmp):
                    #
                    f.append(
                            (
                                sq,
                                data_frames,
                                label_frames,
                                smaple_index,
                                pose_frames,
                                sv_frames,
                                0,
                            )
                        )
            self.files.extend(f)

    def aug(self, a, seed):
        a = np.matmul(a, seed)
        return a[:, :3]

    def __getitem__(self, idx):
        # pos = idx#self.files[idx]
        idxs= []
        seq_idxs = []
        locs = []
        feats = []
        labels_pre = []
        labels_all = []
        poses = []
        reverse = []
        svs = []
        mid_sv = []
        o_coords = []
        sv_coords_indexes = []
        re = []
        # file_list = self.current_data[idx]
        for f in range(len(self.current_data)):
            idxs.append(idx)
            seq_idxs.append(f)
            poses.append(self.current_data[f][4][idx])
            re.append(self.current_data[f][6])
            scan = np.fromfile(self.current_data[f][1][idx], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            coords = scan[:, :3]
            r = scan[:, 3] - np.min(scan[:, 3])
            coords = coords
            o_coords.append(torch.Tensor(np.copy(coords)))
            sample_point = json.load(open(self.current_data[f][3][idx]))
            p_idx = sample_point["0"] + [v for k, v in sample_point.items() if k != "0"]
            sv = np.load(open(self.current_data[f][5][idx], "rb"))
            sv_info = list(np.unique(sv, return_inverse=True, return_counts=True))

            sv_coord = (
                           torch.zeros([sv_info[0].shape[0], 3]).index_add_(
                               0, torch.IntTensor(sv_info[1]).long(), torch.Tensor(coords)
                           )
                       ) / torch.Tensor(sv_info[2]).view(-1, 1)
            sv_coords_indexes.append(
                KDTree(sv_coord.numpy(), leaf_size=100).query(sv_coord.numpy(), k= self.kp, return_distance=False)
            )  # int(sv_coord.shape[0] * 0.6)

            sv_info.append(p_idx)
            svs.append(sv_info)

            coords /= self.scale
            coords = coords.astype(np.int32)

            seq_idx = self.current_data[f][1][idx].split("/")[-3]
            f_idx = self.current_data[f][1][idx].split("/")[-1].split(".")[0]
            mid_tmp = os.path.join(
                self.mid_label, self.mid_data_format.format(seq_idx, f_idx),
            )

            label = np.fromfile(self.current_data[f][2][idx], dtype=np.uint32)
            sem_label = label & 0xFFFF
            for k, v in self.config["learning_map"].items():
                sem_label[sem_label == k] = v
            labels_all.append(torch.Tensor(sem_label.astype(np.int)).view(-1, 1))
            if idx == 0 or not (idx) % self.sample_interval:
                sv_mask = np.isin(sv, p_idx)
                sem_label[(1 - sv_mask).astype(np.bool)] = 0
                for p_i in p_idx:
                    p_lab = np.unique(sem_label[sv == p_i], return_counts=True)
                    if len(p_lab[0]) == 1:
                        continue
                    sem_label[sv == p_i] = p_lab[0][np.argmax(p_lab[1])]
                # sem_label = sem_label.astype(np.int)
            else:
                sem_label = np.zeros_like(sem_label)

            if os.path.isfile(mid_tmp) and idx:
                mid_sem = np.load(open(mid_tmp, "rb"))
                for sv_i, p_i in enumerate(mid_sem[:, 0]):
                    sem_label[sv == p_i] = mid_sem[sv_i, 1]
                mid_sv.append(mid_sem)
            else:
                mid_sv.append(np.array([[-1, -1]]))
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

            locs.append(coords)
            feats.append(r.view(-1, 1))
            labels_pre.append(torch.Tensor(sem_label).view(-1, 1))
            reverse.append(ind[3])
        locs, feats= sparse_collate(locs, feats)
        labels_pre = torch.cat(labels_pre,dim=0)
        return {
            "idx": idxs,
            "seq_idx": seq_idxs,
            "x": [locs, feats],
            "y": labels_pre,
            "y_all": labels_all,
            "reverse_index": reverse,
            "poses": poses,
            "svs": svs,
            "mid_sv": mid_sv,
            "o_coords": o_coords,
            "sv_coords_indexes": sv_coords_indexes,
            "re":re
        }
    def get_data_loader(self,):

        order = list(range(len(self.files)))
        # if self.type == "train":
        #     random.shuffle(order)
        single=False
        f_non_100 = -1
        while order:
            order_idx= order[0]
            if len(self.files[order_idx][1]) ==(self.sample_interval//2+1):

                self.current_data.append( self.files[order_idx])
                order.pop(0)
            else:
                if order_idx == f_non_100 or single:
                    self.current_data.append(self.files[order_idx])
                    order.pop(0)
                    single=True
                else:
                    order.append(order.pop(0))
                    if f_non_100==-1:
                        f_non_100 = order_idx
            if not order:
                break
            if len(self.current_data) == self.batch_size \
                    or (f_non_100 == order[0] and len(self.current_data))\
                    or (single and len(self.current_data)):
                yield torch.utils.data.DataLoader(
                        self, batch_size=1, num_workers=0, shuffle=False,
                    )
                self.current_data = []
        if  len(self.current_data):
            yield torch.utils.data.DataLoader(
                self, batch_size=1, num_workers=0, shuffle=False,
            )
    def train_consist(self, x):
        pass
