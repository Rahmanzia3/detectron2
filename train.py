#  Import detectron libraries
import os
import torch
import numpy as np
from detectron2 import model_zoo
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import HookBase
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import default_setup
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, launch
import wget


'''
DATASET FORMAT IN data_path

    ├── annotations
    ├── images
    ├── names.txt
    ├── test.txt
    └── train.txt

'''
# Add unzip herer


parser = ArgumentParser()
parser.add_argument("--data_source", default= "datasets/licenseplates", help="Where all custom data is stored or a download link")
parser.add_argument("--batch", default=4, help="Batch size")
parser.add_argument("--iterations", default=2000, help="Number of Iterations to run" , type = int)
parser.add_argument("--num_gpus", default=1, help="NUmber of GPus" , type = int)
parser.add_argument("--cfg_model", default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml', help="CFG model")
parser.add_argument("--eval_period", default=50, help="Evaluation iteration number", type = int)
parser.add_argument("--resume_training", default=False, help="Start training from last weights")
parser.add_argument("--project", default='team_alpha_testing', help="Give project name")

opt = parser.parse_args()
current_dir = os.getcwd()
project_dir = os.path.join(current_dir,opt.project)

os.makedirs(project_dir, exist_ok = True)

def download(path):
    if os.path.isdir(path):
        data_path = path
        print('Input data is a directory')
    else:
        test_url =  path.find('https')
        if test_url == 0:
            print('Input is web Source')
            wget.download(path,project_dir)
            # Unzip downloaded data

            list_sub_folder = os.listdir(project_dir)
            for x in list_sub_folder:
                check = x.find('.zip')
                if check != -1:
                    source_zip = os.path.join(project_dir,x)
                    destination_zip = project_dir

                    un_zip(source_zip,destination_zip)
        elif test_url != 0:
            print('Download from google drive')
            # Assign destination zip path
            project_dir_zip = os.path.join(project_dir, 'custom_data.zip')
            download_file_from_google_drive(opt.data_source,project_dir_zip)
            # Unzip Downloaded data
            un_zip(project_dir_zip,project_dir)

        data_path = find_data_folder(project_dir)
    return data_path

def find_name_txt(path):
    sub_dir = os.listdir(path)
    for x in sub_dir:
        if x == 'names.txt':
            names_dir = os.path.join(path,x)


            print('Names dir :',names_dir)
    file1 = open(names_dir,'r')
    lines = file1.readlines()
    class_names = []
    for x in lines:
        x  = x.split("\n",1)[0]
        class_names.append(x)

    return(class_names)

data_path = download(opt.data_source)
CLASS_NAMES = find_name_txt(data_path)

def setup_cfg(opt,path):

    # print(path)


    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.TEST.AUG.ENABLED = True

    cfg.TEST.PRECISE_BN.ENABLED = True

    # cfg = get_cfg()

    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file(opt.cfg_model))

    cfg.DATASETS.TRAIN = ("licenseplates_train",)
    cfg.DATASETS.TEST = ("licenseplates_test",)   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(opt.cfg_model)

    cfg.SOLVER.IMS_PER_BATCH = int(opt.batch)
    cfg.TEST.EVAL_PERIOD = int(opt.eval_period)
    cfg.SOLVER.MAX_ITER = int(opt.iterations)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(len(CLASS_NAMES))

    cfg.freeze()
    # default_setup(cfg, args)

    return cfg

def find_data_folder(source_path):
    sub_folders = os.listdir(source_path)
    for x in sub_folders:
        full_path = os.path.join(source_path,x)
        check_folde = os.path.isdir(full_path)
        if check_folde is True:
            # data fodler
            check_desired_folder = os.listdir(full_path)
            if len(check_desired_folder) >= 2:
                data_folder = full_path

                # print(data_folder)
    return data_folder
def un_zip(source,destination):
        file_name = source
      
        # opening the zip file in READ mode 
        with ZipFile(file_name, 'r') as zip: 
        # printing all the contents of the zip file 
            zip.printdir() 
          
            # extracting all the files 
            print('Extracting all the files now...') 

            # ######## ADD DESTINATION LOCATION HERE
            zip.extractall(destination) 
            print('Done!') 
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        # self.cfg.DATASETS.TRAIN = ('licenseplates_train',)
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

def load_voc_instances(dirname: str, split: str):
    """
    Load licenseplates VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "annotations", "images"
        split (str): one of "train", "test"
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_licenseplates_voc(name, dirname, split):
    DatasetCatalog.register(name,
                            lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES,
                                  dirname=dirname,
                                  split=split)

def main(args):
    # Register licenseplates dataset
    register_licenseplates_voc("licenseplates_train", data_path, "train")
    register_licenseplates_voc("licenseplates_test", data_path, "test")

    # Setup model configuration
    cfg = setup_cfg(opt,data_path)
    # print(cfg)
    # exit()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg) 
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss]) 


    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    if os.path.isdir(opt.resume_training):
        trainer.resume_or_load(resume=opt.resume_training)
    else:
        trainer.resume_or_load(resume=False)

    return trainer.train()

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

#  add unzip

if __name__ == "__main__":

    launch(
        main,
        opt.num_gpus,
        num_machines= 1,
        machine_rank= 0,
        dist_url='tcp://127.0.0.1:50152',
        args=(opt,),
    )








    # parser.add_argument("--download_url", default='/home/tericsoft/team_alpha/all_networks/detectron2/mask/zeta', help="Download link")

    # This is not working  
    # COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml

    # print(opt)
    # print(type(opt))    
