import math
import cv2
import numpy as np
import os
from datetime import datetime as dt
import tarfile
import shutil


def euclidean_dist(p1, p2):
    if len(p1) != len(p2):
        print("warning: computing distance between two points of different dimensions")
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def to_rad(deg):
    return deg * math.pi / 180.0

def in_range(val, rang):
    # Returns True if val is in range (a,b); Inclusive.
    return val >= rang[0] and val <= rang[1]


#### World building ###
def place_object(worldstr, x, y, object_char="r"):
    lines = []
    for line in worldstr.split("\n"):
        if len(line) > 0:
            lines.append(line)
    
    width = len(lines[0])
    length = len(lines)
    lines[y] = lines[y][:x] + object_char + lines[y][x+1:]
    return "\n".join(lines)

def place_objects(worldstr, obj_poses):
    if type(obj_poses) == dict:
        for obj_char in obj_dict:
            x, y = obj_dict[obj_char]
            worldstr = place_object(worldstr, x, y, object_char=obj_char)
    else:
        for obj_char, obj_pose in obj_poses:
            worldstr = place_object(worldstr, obj_pose[0], obj_pose[1], object_char=obj_char)
    return worldstr


#### File Utils ####
def save_images_and_compress(images, outdir, filename="images", img_type="png"):
    # First write the images as temporary files into the outdir
    cur_time = dt.now()
    cur_time_str = cur_time.strftime("%Y%m%d%H%M%S%f")[:-3]    
    img_save_dir = os.path.join(outdir, "tmp_imgs_%s" % cur_time_str)
    os.makedirs(img_save_dir)

    for i, img in enumerate(images):
        img = img.astype(np.float32)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
        img = cv2.rotate(img, cv2.ROTATE_180)  # rotate 90deg clockwise        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        save_path = os.path.join(outdir, img_save_dir, "tmp_img%d.%s" % (i, img_type))
        cv2.imwrite(savepath, img)

    # Then compress the image files in the outdir
    output_filepath = os.path.join(outdir, "%s.tar.gz" % filename)
    with tarfile.open(output_filepath, "w:gz") as tar:
        tar.add(img_save_dir, arcname=filename)

    # Then remove the temporary directory
    shutil.rmtree(img_save_dir)
