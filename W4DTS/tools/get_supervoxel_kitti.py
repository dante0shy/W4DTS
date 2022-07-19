import os, glob, shutil
import tqdm
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor


resolution = 0.6

base_dir = "/path/to/dataset"
data_dir = os.path.join(base_dir,'sequences')
output_dit = os.path.join(base_dir,'superV_v2')

d_format = ""
o_format = "{}/superV/{}.xyz"
on_format = "{}/superV/{}.npy"
tmp_format = "{} {} {} {:d} {:d} {:d}\n"


def trans(p):
    raw = open(p).readlines()
    data = np.array([[float(y) for y in x.rstrip("\n").split(" ")] for x in raw])
    return data


def supervoxel_gen(data):
    s_id = data.split("/")[-3]
    f_id = data.split("/")[-1].split(".")[0]
    o_name = os.path.join(output_dit, o_format.format(s_id, f_id))
    on_name = os.path.join(output_dit, on_format.format(s_id, f_id))
    # if os.path.exists(os.path.join(output_dit, o_format.format(s_id, f_id))):
    #     return
    scan = np.fromfile(data, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # label = np.fromfile(
    #     data.replace("velodyne", "labels").replace(".bin", ".label"), dtype=np.uint32
    # )

    tmp = []
    for i, point in enumerate(scan):
        tmp.append(
            tmp_format.format(
                point[0], point[1], point[2], 0, 0, 0,
            )
        )
    tmp_file_name = "./{}-{}.xyz".format(s_id, f_id)
    open(tmp_file_name, "w").writelines(tmp)
    os.makedirs(
        "/".join(os.path.join(output_dit, o_format.format(s_id, f_id)).split("/")[:-1]),
        exist_ok=True,
    )

    os.system(
        " ".join(
            [
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "superV/supervoxel_vk"
                ),
                tmp_file_name,
                o_name,
                "{}".format(resolution),
            ]
        )
    )
    os.remove(tmp_file_name)
    tmp_data = trans(o_name)

    np.save(open(on_name, "wb"), np.unique(tmp_data[:, 3:], axis=0, return_index=True, return_inverse=True)[2])
    os.remove(o_name)

    pass


if __name__ == "__main__":

    data_list = glob.glob(os.path.join(data_dir, "*", "velodyne", "*"))
    data_list = [x for x in data_list if int(x.split("/")[-3]) < 3]

    with ProcessPoolExecutor(max_workers=4) as excuter:
        excuter.map(supervoxel_gen, data_list)
    # for data in data_list:
    #     supervoxel_gen(data)
    print("finish!")
