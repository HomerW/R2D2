"""
Converts data from the R2D2 raw format to Bridge numpy format.

Consider the following directory structure for the input data:

data/
    success/
        <YYYY>-<MM>-<DD>/ (Local Time)
            <DAY>_<MON>_<DD>_<HH>_<MM>_<SS>_<YYYY>/
                trajectory.h5
                recordings/
                    MP4/
                        <cam_serial_1>.mp4
                        <cam_serial_2>.mp4
                        <cam_serial_3>.mp4
            <DAY>_<MON>_<DD>_<HH>_<MM>_<SS>_<YYYY>/
                trajectory.h5
                recordings/
                    MP4/
                        <cam_serial_1>.mp4
                        <cam_serial_2>.mp4
                        <cam_serial_3>.mp4
            ...
        <YYYY>-<MM>-<DD>/ (Local Time)
            ...
    failure/
        ...

The --depth parameter controls how much of the data to process at the 
--input_path; for example, if --depth=2, then --input_path should be 
"data", and all data will be processed. If --depth=1, then 
--input_path should be "data/success", and all data 
under each date (i.e. 2023-04-17, 2023-04-19, etc.) will be processed.

The same directory structure will be replicated under --output_path.  For 
example, in the second case, the output will be written to 
"{output_path}/<train/val>/...".

Squashes images to 128x128.

Written by Emi Tran and Lawrence Chen
"""
import copy
from functools import partial
import tensorflow as tf
from datetime import datetime
import glob
import os
from collections import defaultdict
from PIL import Image
import numpy as np
from absl import app, flags, logging
import tqdm
import random
from multiprocessing import Pool
from r2d2.trajectory_utils.misc import load_trajectory


FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    1,
    (
        "Number of directories deep to traverse to the dated directory. Looks for"
        "{input_path}/dir_1/dir_2/.../dir_{depth-1}/<YYYY>-<MM>-<DD>/..."
    ),
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float("train_proportion", 0.9, "Proportion of data to use for training (rather than val)")
flags.DEFINE_integer("num_workers", 1, "Number of threads to use")


# Camera ID's # From the laptop perspective
# Use zed.get_camera_information().serial_number to find out the following
hand_camera_id = "13062452"  # Hand camera (on gripper)
varied_camera_1_id = "24259877"  # Left camera
varied_camera_2_id = "20521388"  # Right camera
camera_names = [
    hand_camera_id + "_left",
    hand_camera_id + "_right",
    varied_camera_1_id + "_left",
    varied_camera_1_id + "_right",
    varied_camera_2_id + "_left",
    varied_camera_2_id + "_right",
]


def squash(im):  # squash from 480x640 to 128x128 and flattened as a tensor
    im = Image.fromarray(im)
    im = im.resize((128, 128), Image.LANCZOS)
    out = np.asarray(im)
    return out.astype(np.uint8)


def process_images(all_traj):  # processes images at a trajectory level
    images_out = defaultdict(list)

    missing_cam_traj = []
    for i, traj in enumerate(all_traj):
        image = traj["observation"]["image"]
        if set(camera_names) != set(image.keys()):
            missing_cam_traj.append(i)
            continue
        for i, cam in enumerate(camera_names):
            # BGRD -> RGB (ignore depth channel)
            images_out[f"images{i}"].append(squash(image[cam][:-1][..., ::-1]))
    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for k in images_out.keys():
        obs[k] = images_out[k][:-1]
        next_obs[k] = images_out[k][1:]
    return obs, next_obs, missing_cam_traj


def process_actions(all_traj, skips):
    out = []
    for i, traj in enumerate(all_traj):
        if i in skips:
            continue
        cartesian_pos = np.array(traj["action"]["cartesian_position"])
        gripper_pos = traj["action"]["gripper_position"]
        cartesian_pos = np.append(cartesian_pos, gripper_pos)
        out.append(cartesian_pos)
    return np.array(out[:-1])  # Exclude the last action


def process_state(all_traj, skips):
    out = []
    for i, traj in enumerate(all_traj):
        if i in skips:
            continue
        cartesian_pos = np.array(traj["observation"]["robot_state"]["cartesian_position"])
        gripper_pos = traj["observation"]["robot_state"]["gripper_position"]
        cartesian_pos = np.append(cartesian_pos, gripper_pos)
        out.append(cartesian_pos)
    out = np.array(out)
    return out[:-1], out[1:]


def process_time(all_traj, skips):
    out = []
    for i, traj in enumerate(all_traj):
        if i in skips:
            continue
        time = np.array(traj["observation"]["timestamp"]["robot_state"]["read_start"])
        out.append(time)
    out = np.array(out)
    return out[:-1], out[1:]


# processes each data collection attempt
def process_dc(path, train_ratio=0.9):
    all_dicts_train = list()
    all_dicts_test = list()
    all_rews_train = list()
    all_rews_test = list()

    # Get all "<DAY>_<MON>_<DD>_<HH>_<MM>_<SS>_<YYYY>/"
    # under "<YYYY>-<MM>-<DD>/ (Local Time)"
    search_path = os.path.join(path, "*")
    all_traj = glob.glob(search_path)
    if all_traj == []:
        logging.info(f"no trajs found in {search_path}")
        return [], [], [], []

    random.shuffle(all_traj)

    num_traj = len(all_traj)
    for itraj, tp in tqdm.tqdm(enumerate(all_traj)):
        try:
            out = dict()
            ld = os.listdir(tp)

            assert "trajectory.h5" in ld, tp + ":" + str(ld)
            assert "recordings" in ld, tp + ":" + str(ld)

            traj_batch = load_trajectory(
                filepath=os.path.join(tp, "trajectory.h5"), 
                recording_folderpath=os.path.join(tp, "recordings/MP4")
            )

            # Process image will get all image. Also return a set of trajectory index to ignore
            # Because sometimes not all 6 cameras are working, so we will ignore that trajectory
            obs, next_obs, skips = process_images(traj_batch)

            acts = process_actions(traj_batch, skips)
            state, next_state = process_state(traj_batch, skips)
            time_stamp, next_time_stamp = process_time(traj_batch, skips)
            term = [0] * len(acts)
            term[-1] = 1  # Last element is 1 to flag trajectory's end

            out["observations"] = obs
            out["observations"]["state"] = state
            out["observations"]["time_stamp"] = time_stamp
            out["next_observations"] = next_obs
            out["next_observations"]["state"] = next_state
            out["next_observations"]["time_stamp"] = next_time_stamp
            # Obs: images0, images1,  ..., state, time_stamp,

            out["observations"] = [
                dict(zip(out["observations"], t)) for t in zip(*out["observations"].values())
            ]
            out["next_observations"] = [
                dict(zip(out["next_observations"], t)) for t in zip(*out["next_observations"].values())
            ]

            out["actions"] = acts
            out["terminals"] = term

            labeled_rew = copy.deepcopy(out["terminals"])[:]
            labeled_rew[-2:] = [1, 1]

            traj_len = len(out["observations"])
            assert len(out["next_observations"]) == traj_len
            assert len(out["actions"]) == traj_len
            assert len(out["terminals"]) == traj_len
            assert len(labeled_rew) == traj_len

            if itraj < int(num_traj * train_ratio):
                all_dicts_train.append(out)
                all_rews_train.append(labeled_rew)
            else:
                all_dicts_test.append(out)
                all_rews_test.append(labeled_rew)
        except FileNotFoundError as e:
            logging.error(e)
            continue
        except AssertionError as e:
            logging.error(e)
            continue

    return all_dicts_train, all_dicts_test, all_rews_train, all_rews_test


def make_numpy(path, train_proportion):
    dirname = os.path.abspath(path)
    outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-(max(FLAGS.depth - 1, 1)) :])

    if os.path.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)

    lst_train = []
    lst_val = []
    rew_train_l = []
    rew_val_l = []

    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val = process_dc(
            os.path.join(path, dated_folder), train_ratio=train_proportion
        )

        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)

    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, lst_train)
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, lst_val)

    # doesn't seem like these are ever used anymore
    # np.save(os.path.join(outpath_train, "out_rew.npy"), rew_train_l)
    # np.save(os.path.join(outpath_val, "out_rew.npy"), rew_val_l)


def main(_):
    assert FLAGS.depth >= 1

    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))

    worker_fn = partial(make_numpy, train_proportion=FLAGS.train_proportion)

    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(worker_fn, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
