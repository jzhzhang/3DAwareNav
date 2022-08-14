# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T

from constants import coco_categories_mapping
from skimage import io
from scipy.special import softmax

class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.args = args

    def get_prediction(self, img, depth, cat_goal):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2)

        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros((img.shape[0], img.shape[1], 6 + 1))
        # semantic_probability = np.zeros((img.shape[0], img.shape[1], 6 + 1))

        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):

            import time
            t_s = time.time() 

            if class_idx in list(coco_categories_mapping.keys()):
                # print("class_idx", class_idx)
                # idx = coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]['instances'].pred_masks[j] * 1.
                # semantic_input[:, :, idx] += obj_mask.cpu().numpy()
 
                #---------------------- semantic probs ---------------------- #
                tmp_semantic_probability = np.zeros((img.shape[0], img.shape[1], 6 + 1))

                obj_all_class_scores = seg_predictions[0]['instances'].all_class_scores[j].cpu().numpy() # shape=80

                #non_zero_index = np.nonzero(obj_mask.cpu().numpy())
                sum_prob_ours = 0.0
                for key in coco_categories_mapping.keys():
                    index_coco = int(key)
                    index_ours = int(coco_categories_mapping[key])
                    tmp_semantic_probability[:,:,index_ours] += obj_all_class_scores[index_coco]
                    sum_prob_ours += obj_all_class_scores[index_coco]
                tmp_semantic_probability[:,:,-1] += np.sum(obj_all_class_scores) - sum_prob_ours # sum of other class score
                #tmp_semantic_probability[:,:,-1] += 1 - sum_prob_ours # sum of other class score and bg

                sum_normalize = np.sum(tmp_semantic_probability, axis=-1)
                tmp_semantic_probability /= sum_normalize[...,None] # Normalize

                # print(obj_mask.cpu().numpy().shape) #480*640
                mask_filter = np.where(obj_mask.cpu().numpy()==0) #tuple, index of zeros
                tmp_semantic_probability[mask_filter] = 0

                # print("--------------------------------")
                # print(tmp_semantic_probability[0,0,:])
                # print(semantic_probability[0,0,:])
                # print("--------------------------------")

                semantic_input += tmp_semantic_probability
            # print(time.time() - t_s)
            #print(cut)
                #---------------------- semantic probs ---------------------- #


        # print("xxxx",semantic_input[0, 0, :])
        # semantic_input = semantic_probability

        # Normalize for Overlap bbox
        overlap_index = np.where(np.sum(semantic_input, axis=-1)>1)
        semantic_input[overlap_index] /= np.sum(semantic_input[overlap_index], axis=-1).reshape(-1,1)
        
        # make bg as grey
        semantic_input[:,:,-1] = 1 - np.sum(semantic_input[:,:,:-1], axis=-1) # only make bk_prob = 1

        #semantic_pred = torch.nn.functional.softmax(seg_predictions[0]['sem_seg'], dim=0).permute(1,2,0).cpu().numpy()
        # semantic_pred = semantic_input
        # entropy_tmp = -semantic_pred*np.log(semantic_pred)
        # sem_entropy = np.sum(entropy_tmp, axis=2)

        sem_entropy = np.zeros((semantic_input.shape[0], semantic_input.shape[1]))
        #=================prob points================
        goal_cat_output = semantic_input[:, :, cat_goal] + 1e-5
        goal_cat_output[goal_cat_output<0] = 0



        return semantic_input, sem_entropy, goal_cat_output


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map

# config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml
# config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
class ImageSegmentation():
    def __init__(self, args):
        string_args = """
            --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
            --input input1.jpeg
            --confidence-threshold {}
            --opts MODEL.WEIGHTS
            ./weight/model_final_c10459.pkl
            """.format(args.sem_pred_prob_thr, args.semantic_weight)


        # string_args = """
        #     --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        #     --input input1.jpeg
        #     --confidence-threshold {}
        #     --opts MODEL.WEIGHTS
        #     detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
        #     """.format(args.sem_pred_prob_thr)


        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE {}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return all_predictions, vis_output


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        # print("dataset", self.metadata.thing_classes)
        # print("dataset", self.metadata.stuff_classes)
        # print("dataset", self.metadata.panoptic_root)
        # print("dataset", self.metadata.ignore_label)



        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
