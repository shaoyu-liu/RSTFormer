"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.eval import DVSEvaluator
from torch.utils.data import ConcatDataset
from datasets import build_dataset
from datasets.data_prefetcher import data_prefetcher
import bisect
import json

class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], (dataset_idx, idx)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_type: str = 'spatial', input_type: str = 'frame_only'):
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, input_type, prefetch=True)
    samples, events, targets, seq_indexes, _ = prefetcher.next()
    

    pre_states_aps = [None] * 4
    pre_states_dvs = [None] * 4

    for _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
        outputs,pre_states_aps,pre_states_dvs = model(samples, events, pre_states_aps, pre_states_dvs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()

        pre_states_aps = Todetach(pre_states_aps)
        pre_states_dvs = Todetach(pre_states_dvs)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, events, targets, seq_indexes, _ = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, base_ds, data_loader, device, model_type: str = 'temporal',
             input_type: str = 'multi-modal'):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    dvs_evaluator = DVSEvaluator(iou_types, base_ds)
    # prefetcher = data_prefetcher(data_loader, device, input_type, prefetch=True)
    # samples, events, targets, seq_indexes, _ = prefetcher.next()

    pre_states_aps = [None] * 4
    pre_states_dvs = [None] * 4

    for inputs, indexes in metric_logger.log_every(data_loader, 10, header):
        samples, events, targets = inputs
        seq_indexes, img_ids = indexes
        samples = samples.to(device) if input_type == 'frame_only' or input_type == 'multi-modal' else samples
        events = events.to(device) if input_type == 'event_only' or input_type == 'multi-modal' else events
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        unimodal_input = samples if input_type == 'frame_only' else events  

        outputs,pre_states_aps,pre_states_dvs = model(samples, events, pre_states_aps, pre_states_dvs)   
        #outputs = model(samples, events)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, 'test')

        res = [(img_id, output) for img_id, output in zip(img_ids, results)]

        if dvs_evaluator is not None:
            dvs_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    if dvs_evaluator is not None:
        dvs_evaluator.synchronize_between_processes()
        dvs_evaluator.accumulate()
        dvs_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    with open("./results/models/SODFormer/train/condition-detr-test.txt", "a") as f:
                f.write(json.dumps(stats) + "\n")
   
    if 'bbox' in postprocessors.keys():
        stats['coco_eval_bbox'] = dvs_evaluator.coco_eval['bbox'].stats.tolist()
        
        with open("./results/models/SODFormer/train/eval_metrics.txt", "a") as f_metrics:
            f_metrics.write("IoU metric: bbox\n")
            metrics = [
                ['AP', '0.50:0.95', '', 'all'],
                ['AP', '0.50', '', 'all'],
                ['AP', '0.75', '', 'all'],
                ['AP', '0.50:0.95', '', 'small'],
                ['AP', '0.50:0.95', '', 'medium'],
                ['AP', '0.50:0.95', '', 'large']
            ]

            for i, (metric_name, iou_range, _, area) in enumerate(metrics):
                ap = dvs_evaluator.coco_eval['bbox'].stats[i]
                f_metrics.write(f"Average Precision  ({metric_name}) @[ IoU={(iou_range if iou_range != '' else '0.50')} | area={area} ] = {ap:.3f}\n")

    return dvs_evaluator


def concatenate_dataset(image_set, args):
    set_dict = {
        'train':132,
        'val':44,
        'test':44
    }
    entire_dataset = []
    subdataset_num = set_dict[image_set]
    print('Loading {} sub-datasets'.format(image_set))
    base_folder = os.path.join(args.frame_path, image_set)

    scene_list = os.listdir(base_folder)
    pro = 0
    for scene in scene_list:
        scene_folder = os.path.join(base_folder, scene)
        sub_datasets = os.listdir(scene_folder)
        for dataset_name in sub_datasets:
            print('\r', end='')
            print('Loading Progress: {:.2%}'.format(pro / subdataset_num), '▋' * (pro * 50 // subdataset_num), end='')
            sys.stdout.flush()
            entire_dataset.append(build_dataset(image_set, args, scene, dataset_name))
            pro += 1

    print('\n')
    return ConcatDatasetCustom(entire_dataset)

def Todetach(prev_states):
    detacher_nested_list = []
    for inner_list in prev_states:
        detached_inner_list = []
        for tensor in inner_list:
            detached_tensor = tensor.detach()
            detached_inner_list.append(detached_tensor)
        detacher_nested_list.append(detached_inner_list)
    prev_states = detacher_nested_list
    return prev_states 