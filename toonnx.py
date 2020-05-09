from __future__ import division

from models.darknet import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.yolo_nano_helper import YoloNano
from torch.nn.parallel import DataParallel
from torch.onnx import OperatorExportTypes

torch.set_default_tensor_type(torch.FloatTensor)

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)

@torch.no_grad()
def evaluate(model, path, conf_thres, nms_thres, img_size, batch_size,data_type):
    T_ious = np.linspace(0.5,0.95,num=10)
    # Get dataloader
    #img_size=416,augment=False,use_mix=True, multiscale=True, normalized_labels=True,data_type="coco_train"
    dataset = ListDataset(path, img_size=img_size, augment=False,use_mix=False,multiscale=False,normalized_labels=True,data_type =data_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    labels = []
    sample_metrics=[]
    for batch_i, (_, imgs, targets,img_scores,gt_mix_index) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        if targets is None:
            continue
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        outputs = model(imgs)

        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets,T_ious)


    true_positives, pred_scores, pred_labels = [x for x in list(zip(*sample_metrics))]
    true_positives = np.concatenate(true_positives,1)
    pred_scores = np.concatenate(pred_scores,0)
    pred_labels = np.concatenate(pred_labels,0)

    precision = [0. for i in range(len(T_ious))]
    recall = [0. for i in range(len(T_ious))]
    AP = [0. for i in range(len(T_ious))]
    f1 = [0. for i in range(len(T_ious))]
    ap_class = [0. for i in range(len(T_ious))]
    for _,true_positive in enumerate(true_positives):
        precision[_], recall[_], AP[_], f1[_], ap_class[_] = ap_per_class(true_positive, pred_scores, pred_labels, labels,T_ious[_])

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolo-nano_person.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=208, help="size of each image dimension")
    parser.add_argument("--data_type", type=str, default="coco_test", help="Dataset type")

    # yolo_anno = YoloNano(1,80)
    # print(yolo_anno)

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)


    valid_path = data_config["valid"]

    class_names = load_classes(data_config["names"])

    if 'yolov3' in opt.model_def:
        model = Darknet(opt.model_def).to(device)
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    else:
        kargs = get_nano_info(opt.model_def)
        kargs['to_onnx'] = True
        rmodel = YoloNano(**kargs).to(device)
        rmodel.apply(weights_init_normal)
        model = WrappedModel(rmodel)
        #model = DataParallel(model)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    model_name = opt.model_def.split('.')[0].split('/')[-1]
    dummy_input = torch.randn(1, 3, opt.img_size, opt.img_size, device=device)
    input_names = ["image_input"]
    output_names = ["yolo_map_%d"%d for d in (1,2, 3) ]
    torch.onnx.export(rmodel, dummy_input, "%s.onnx"%model_name, verbose=True, input_names=input_names, output_names=output_names,do_constant_folding=True, operator_export_type=OperatorExportTypes.ONNX, aten=False, opset_version=11, keep_initializers_as_inputs=True)