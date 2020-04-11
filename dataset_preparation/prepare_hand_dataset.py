import os
from os import path as osp
import argparse
from shutil import copy as scp

from scipy.io import loadmat
import numpy as np
np.random.seed(0)
import cv2

'''
The data is structured as follows: The folder "_LABELLED_SAMPLES" contains 48 folders, one for each
video. The name of each folder is ACTIVITY_LOCATION_VIEWER_PARTNER, where:
 - ACTIVITY = {CARDS, CHESS, JENGA, PUZZLE}
 - LOCATION = {COURTYARDS, OFFICE, LIVINGROOM}
 - VIEWER/PARTNER = {B, H, S, T}

Each folder contains 100 frames as JPEG files and an anotation file as "polygons.mat". 
One anotation file contains polygons with hand segmentation of each person in frames with shape (1, 100). 
100 presents the number of frames where each frame contains a tuple of 4 np.narray with a following order: 
"own left", "own right", "other left", and "other right" hand respectively. 
If one hand dont appear in a certain frame, the np.narray will be empty.
Otherwise, np.narray include n rows which each row store a point [x, y] of particular hand region.

All ground-truth data is stored in metadata.mat file
in the root directory. This file contains 4 fields: '__header__', '__version__', '__globals__', 'video'. 
'video' field contains a MATLAB struct array, which has the
following fields:
 - video_id
 - partner_video_id
 - ego_viewer_id
 - partner_id
 - location_id
 - activity_id
 - labeled_frames (array)  ---  frame_ids
                           ---  polygons with hand segmentations


'''

def list_image_paths_in_dir(base_path, dir):
    '''
    List image paths in a directory

    Args:
    ---
    - base_path: base path of data
    - dir: a particular directory of images

    Returns:
    ---
    - image_path_array (list): a list of image paths in given dir.
    '''
    image_path_array = []
    for root, dirs, filenames in os.walk(base_path + '/' + dir):
        for f in filenames:
            if(f.split(".")[1] == "jpg"):
                img_path = osp.join(root, f)
                image_path_array.append(img_path)

    #sort image_path_array to ensure its in the low to high order expected in polygon.mat
    image_path_array.sort()

    return image_path_array

def copy_rename_files(base_path, dst_path="data/custom/images"):
    '''
    Copy and Rename all image files in base_path to store in dst_path
    '''
    loop_index = 0
    for root, dirs, filenames in os.walk(base_path):
        for dir in dirs:
            for f in os.listdir(osp.join(base_path, dir)):
                if (dir not in f):
                    if(f.split(".")[1] == "jpg"):
                        loop_index += 1
                        new_name = dir + "_" + f
                        print(new_name)
                        scp(osp.join(root, dir, f), osp.join(dst_path, new_name))
                else:
                    break

def get_bbox_visualize(base_path, dir, dest_images_dir, data_file_obj, dest_label_dir):
    '''
    Get bounding boxs and visualize them on frames to get more intuition.
    Also, save these bboxs to label_file with following [label_idx, x_center, y_center, box_width, box_height] on each single line.
    All coordinates must be scaled to [0, 1] and the label_idx should be zero-indexed which corresponds to the row of classes.names.
    Besides, each image in data_file should include appropriate a label_file. 

    Args:
    ---
    - base_path: base path of data
    - dir: a particular directory of images
    - dest_images_dir: destination directory of images folder
    - data_file_obj: file object to save image paths
    - dest_label_dir: destination dir to store label files. 
    '''
    image_path_array = list_image_paths_in_dir(base_path, dir)
    # print(image_path_array)

    boxes = loadmat(osp.join(base_path, dir, "polygons.mat"))

    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    # frame_poly = polygons[0]
    # print(len(frame_poly))
    pointindex = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_poly in polygons:
        index = 0

        img_id = image_path_array[pointindex] # images in base_path
        img = cv2.imread(img_id)

        # img_params = {}
        # img_params["width"] = np.size(img, 1)
        # img_params["height"] = np.size(img, 0)
        head, tail = os.path.split(img_id)
        # video_name = osp.split(head)[1]
        # img_params["filename"] = tail
        # img_params["path"] = os.path.abspath(img_id)
        # img_params["type"] = "train"

        frame_label_file = dest_label_dir + '/' + dir + '_' + tail.split('.')[0] + '.txt'
        print("label_file:", frame_label_file)

        # Open frame_label_file
        label_file_obj = open(frame_label_file, "wt")

        img_height, img_width = img.shape[:-1]

        pointindex += 1

        boxarray = []
        # csvholder = []
        for pointlist in frame_poly:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = height = width = 0

            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    # Find 4 point of bbox
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    # print(index, "====", len(point))

                    # Store point of polygons for plotting
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

            # hold = {}
            # hold['minx'] = min_x
            # hold['miny'] = min_y
            # hold['maxx'] = max_x
            # hold['maxy'] = max_y
            
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                box_w = (max_x - min_x)
                box_h = (max_y - min_y)

                labelrow = [0, center_x/img_width, center_y/img_height, box_w/img_width, box_h/img_height]

                # Save labelrow to label_file
                label_file_obj.write(f"{labelrow[0]} {labelrow[1]} {labelrow[2]} {labelrow[3]} {labelrow[4]}\n")
                # Save img_id to data_file
                data_file_obj.write(dest_images_dir+'/'+dir+'_'+tail+'\n')


            cv2.polylines(img, [pst], True, (0, 255, 255), 1)
            cv2.rectangle(img, (min_x, max_y),
                          (max_x, min_y), (0, 255, 0), 1)

        cv2.putText(img, "DIR : " + dir + " - " + tail, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
        cv2.imshow('Verifying annotation ', img)
        cv2.waitKey(1)  # close window when a key press is detected

        # Close label_file
        label_file_obj.close()

def split_dataset_x_save_files(base_path, dst_path):
    '''
    Split image dataset into train/val/test base on number of videos. 
    Particularly, we randomly split videos with number of videos in train, val and test respectively are 36, 4 and 8.
    Furthermore, we save image paths as well as labels of each set to file as text file (.txt). 
    Image paths and labels of sets are stored in dst_path/ and dst_path/labels respectively. 
    Name of files are same as set name (e.g. train.txt, val.txt, test.txt, ...)

    Args:
    ---
    - base_path: base path of data
    - dst_path: destination path to save image paths and labels belonging to each set. 
    '''
    folder_paths = []
    for root, dirs, f in os.walk(base_path):
        for dir in dirs:
            print(dir)
            folder_paths.append(dir)
            # for f in os.listdir(osp.join(base_path, dir)):
            #     if(f.split(".")[1] == "jpg"):
            #         scp(osp.join(root, dir, f), osp.join(dst_path, new_name))

    print("Length of folder_paths: %d" %len(folder_paths))

    folder_paths = np.array(folder_paths)

    total_videos = len(folder_paths)
    print("Total videos: %d"%total_videos)

    num_train_videos = 36
    num_val_videos = 4
    num_test_videos = 8

    shuffe_indices = np.arange(total_videos)
    np.random.shuffle(shuffe_indices)
    # print(shuffe_indices)
    train_videos = (folder_paths[shuffe_indices[:num_train_videos]]).tolist()
    val_videos = (folder_paths[shuffe_indices[num_train_videos:-num_test_videos]]).tolist()
    test_videos = (folder_paths[shuffe_indices[-num_test_videos:]]).tolist()

    print("Number of Train, Val, Test videos: %d, %d, %d" %(len(train_videos), len(val_videos), len(test_videos)))
    # print(shuffe_indices[:num_train_videos])
    # print(shuffe_indices[num_train_videos:-num_test_videos])
    # print(shuffe_indices[-num_test_videos:])

    video_dataset = {'train': train_videos,
                    'valid': val_videos,
                    'test': test_videos}

    dst_image_dir = dst_path + 'images'
    dst_label_dir = dst_path + 'labels'
    for name_set, video_list in video_dataset.items():
        data_file = dst_path + name_set + '.txt'

        print("data_file:", data_file)

        data_file_obj = open(data_file, "wt")

        for video_name in video_list:
            print("Video_name:", video_name)
            get_bbox_visualize(base_path, video_name, dst_image_dir, data_file_obj, dst_label_dir)

        # Close files
        data_file_obj.close()


def main(args):
    # metadata = loadmat(r"C:\Users\User\Desktop\Hand dataset\metadata.mat")
    # print("Meta data keys: ", metadata.keys())
    # print(metadata['__header__'])
    # print(metadata['__version__'])
    # print(metadata['__globals__'])
    # print((metadata['video'][0, 0]))

    # anotation = loadmat(r'C:\Users\User\Desktop\Hand dataset\_LABELLED_SAMPLES\CARDS_COURTYARD_B_T\polygons.mat')
    # print(len(anotation['polygons'][0]))

    dst_images_dir = args.dest_dir + 'images'

    # copy_rename_files(args.data_dir, dst_images_dir)

    split_dataset_x_save_files(args.data_dir, args.dest_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand dataset preparation")
    parser.add_argument("-ddir", "--data_dir", type=str, default="",\
            help="Data directory")
    parser.add_argument("-dstdir", "--dest_dir", type=str, default="data/custom/",\
            help="Destination directory of custom data")
    
    args = parser.parse_args()

    main(args)


