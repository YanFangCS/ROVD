import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import json
import logging
from detectron2.structures import BoxMode


logger = logging.getLogger(__name__)
_ROVD_DATASET_SPLIT = {
    "rovd_train":("train", "rovd_train_file_list.txt"),
    "rovd_test":("test", "rovd_test_file_list.txt"), 
    "rovd_train_labeled":("train", "rovd_train_labeled_file_list.txt"),
    "rovd_train_unlabeled":("train", "rovd_train_unlabeled_file_list.txt"),
}


def get_rovd_data_dicts(split, data_root, flist_name=None):
    timer = Timer()
    if split == 'test':
        img_dir = os.path.join(data_root, 'test', 'image')
        ann_dir = os.path.join(data_root, 'test', 'annotations_d2')
    else:
        img_dir = os.path.join(data_root, 'train', 'image')
        ann_dir = os.path.join(data_root, 'train', 'annotations_d2')
        
    with open(flist_name, 'r') as f:
        f_list = f.readlines()
        f_list = [t.strip() for t in f_list]
    
    dataset_dicts = []
    for f_dir in f_list:
        jsns = os.listdir(os.path.join(ann_dir, f_dir))
        for jsn in jsns:
            with open(os.path.join(ann_dir, f_dir, jsn)) as f:
                record = json.load(f)
            record['annotations'] = record['annotations']
            for t in record['annotations']:
                t['bbox_mode']=BoxMode.XYXY_ABS
            # record.pop('annotations')
            if len(record['annotations']) < 1:
                continue
            dataset_dicts.append(record)
            
    if timer.seconds() > 1:
        logger.info(
            "Loading {} data for {} takes {:.2f} seconds.".format(len(dataset_dicts), split, timer.seconds())
        )        
    
    logger.info("Loaded {} images in Detectron2 format from {}".format(len(dataset_dicts), ann_dir))

    return dataset_dicts

def get_rovd_unlabeled_data_dicts(data_root, flist_name=None, labeled=True,):
    timer = Timer()

    img_dir = os.path.join(data_root, 'train', 'image')
    ann_dir = os.path.join(data_root, 'train', 'annotations_d2')
        
    with open(flist_name, 'r') as f:
        f_list = f.readlines()
        f_list = [t.strip() for t in f_list]
    
    dataset_dicts = []
    for f_dir in f_list:
        jsns = os.listdir(os.path.join(ann_dir, f_dir))
        for jsn in jsns:
            with open(os.path.join(ann_dir, f_dir, jsn)) as f:
                record = json.load(f)
            new_record = {
                'file_name':record['file_name'],
                'height':record['height'],
                'width':record['width'],
            }
            dataset_dicts.append(new_record)
            
    if timer.seconds() > 1:
        logger.info(
            "Loading {} data for unlabeled training sets takes {:.2f} seconds.".format(len(dataset_dicts), timer.seconds())
        )        
    
    logger.info("Loaded {} images in Detectron2 format from {}".format(len(dataset_dicts), ann_dir))

    return dataset_dicts
    
def register_rovd_instances(name, split, data_root, flist_name, labeled, metadata,):
    if labeled:
        DatasetCatalog.register(
            name, lambda:get_rovd_data_dicts(split, data_root=data_root, flist_name=flist_name)
        )
    else:
        DatasetCatalog.register(
            name, lambda:get_rovd_unlabeled_data_dicts(data_root=data_root, flist_name=flist_name, labeled=False)
        )
    
    MetadataCatalog.get(name).set(
        evaluator_type='coco', **metadata
    )

def register_rovd_datasets(root):
    with open(os.path.join(root, "ROVD", "category_list.txt"), 'r') as f:
        classes_list = f.readlines()
        classes_list = [t.strip() for t in classes_list]
    for key, (split, flist_name) in _ROVD_DATASET_SPLIT.items():
        meta = {'thing_classes':classes_list}
        labeled = bool('unlabeled' not in key)
        register_rovd_instances(key, split, data_root=root, flist_name=flist_name, labeled=labeled, metadata=meta)
    
_root = os.getenv("DETECTRON2_DATASETS","dataset")
register_rovd_datasets(_root)

if __name__ == "__main__":
    for key, (split, flist_name) in _ROVD_DATASET_SPLIT.items():
        data = DatasetCatalog.get(key)
        print(data)