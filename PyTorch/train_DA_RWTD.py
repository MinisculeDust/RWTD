import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import models_DA
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize, DepthNorm
import time

PROJECT = "DA-RWTD"


def is_rank_zero(args):
    return args.rank == 0


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    if args.architecture == 'DA_RWTD':
        model = models_DA.DA_RWTD.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                                 norm=args.norm)

        from models_DA.Discriminator import FocalLoss, WTD
        Dis = WTD(class_num=2)
    elif args.architecture == 'DA_Dis':
        model = models_DA.DA_RWTD.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                                 norm=args.norm)

        from models_DA.Discriminator import FocalLoss, WTD
        Dis = FocalLoss(class_num=2)
    elif args.architecture == '2DA_Dis':
        model = models_2DA_Dis.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                              norm=args.norm)

        from models_2DA_Dis.FocalLoss import FocalLoss
        Dis = FocalLoss(class_num=2)
    elif args.architecture == '2Dis':
        model = models_2Dis.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                              norm=args.norm)

        from models_2Dis.FocalLoss import FocalLoss
        Dis1 = FocalLoss(class_num=2)
        Dis2 = FocalLoss(class_num=2)
    elif args.architecture == 'DA':
        model = models_DA.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                              norm=args.norm)
    elif args.architecture == 'Normal':
        model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                              norm=args.norm)
    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False

    if args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    if args.architecture == 'DA_RWTD' or args.architecture == 'DA_Dis':
        trained_model, Dis = train(model, Dis, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
              experiment_name=args.name, optimizer_state_dict=None)
    elif args.architecture == '2DA_Dis':
        trained_model, Dis, Dis1 = train(model, Dis, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
              experiment_name=args.name, optimizer_state_dict=None)

    elif args.architecture == '2Dis':
        trained_model, Dis, Dis1 = train(model, Dis2, args, Dis1=Dis1, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
              experiment_name=args.name, optimizer_state_dict=None)

    elif args.architecture == 'DA':
        trained_model = train(model, None, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
              experiment_name=args.name, optimizer_state_dict=None)


    # save the trained-model
    save_path = args.tm + \
                'Ubuntu_' + \
                time.strftime("%Y_%m_%d-%H_%M_UK") \
                + '_epochs-' + str(args.epochs) \
                + '_lr-' + str(args.lr) \
                + '_bs-' + str(args.bs) \
                + '_maxDepth-' + str(args.comments) \
                + '.pth'
    torch.save(trained_model.state_dict(), save_path)
    print('saved the trained model to:', save_path)

    # save trained Dis
    if args.architecture == 'DA_RWTD' or args.architecture == 'DA_Dis':
        save_path_Dis = args.tm + \
                    'Ubuntu_' + \
                    time.strftime("%Y_%m_%d-%H_%M_UK") \
                    + '_epochs-' + str(args.epochs) \
                    + '_lr-' + str(args.lr) \
                    + '_bs-' + str(args.bs) \
                    + '_maxDepth-' + str(args.comments) \
                    + '_Dis.pth'
        torch.save(Dis.state_dict(), save_path_Dis)
        print('saved the Dis model to:', save_path_Dis)
    elif args.architecture == '2Dis':
        save_path_Dis1 = args.tm + \
                       'Ubuntu_' + \
                       time.strftime("%Y_%m_%d-%H_%M_UK") \
                       + '_epochs-' + str(args.epochs) \
                       + '_lr-' + str(args.lr) \
                       + '_bs-' + str(args.bs) \
                       + '_maxDepth-' + str(args.comments) \
                       + '_Dis1.pth'
        torch.save(Dis.state_dict(), save_path_Dis1)
        save_path_Dis2 = args.tm + \
                        'Ubuntu_' + \
                        time.strftime("%Y_%m_%d-%H_%M_UK") \
                        + '_epochs-' + str(args.epochs) \
                        + '_lr-' + str(args.lr) \
                        + '_bs-' + str(args.bs) \
                        + '_maxDepth-' + str(args.comments) \
                        + '_Dis2.pth'
        torch.save(Dis.state_dict(), save_path_Dis2)
        print('saved the Dis model to:', save_path_Dis1, ' and \n', save_path_Dis2)

def train(model, Dis, args, Dis1=None, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    logging = args.logging
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = args.rank == 0
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    train_t_loader = DepthDataLoader(args, 'target').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    NLL_criterion = torch.nn.NLLLoss() # for domain labels
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    # Train the same number of batches from both datasets
    max_batches = min(len(train_loader), len(train_t_loader))

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ############################################################
        #                   Train loop
        # ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)

        # iteratively read the dataloader
        train_source_iter = iter(train_loader)
        train_target_iter = iter(train_t_loader)


        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            # load batches one by one
            source_sample_batch = next(train_source_iter)
            if (i+1) % len(train_t_loader) == 0:
                train_target_iter = iter(train_t_loader)
            target_sample_batch = next(train_target_iter)

            optimizer.zero_grad()

            ###############################################################
            ###                 Load Source Domain Data                 ###
            ###############################################################
            img_s = source_sample_batch['image'].to(device)
            depth_s = source_sample_batch['depth'].to(device)
            # transpose the tensor
            depth_s = depth_s.permute(0, 3, 1, 2)

            # remove outliers and do depth estimation
            # depth = DepthNorm(depth, minDepth=args.min_depth_eval, maxDepth=args.max_depth_eval, doNorm=False)
            if 'has_valid_depth' in source_sample_batch:
                if not source_sample_batch['has_valid_depth']:
                    continue

            ###############################################################
            ###                 Load Target Domain Data                 ###
            ###############################################################
            img_t = target_sample_batch['image'].to(device)

            if 'has_valid_depth' in target_sample_batch:
                if not target_sample_batch['has_valid_depth']:
                    continue

            ###############################################################
            ###           Training progress and GRL lambda              ###
            ###############################################################
            p = float(i + epoch * max_batches) / (args.epochs * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1



            ###############################################################
            ###                  Train on Target Domain                 ###
            ###############################################################
            y_t_domain = torch.zeros(len(img_t), dtype=torch.long).to(device)

            # bin_edges --> [8, 81]
            # pred --> torch.Size([8, 1, 128, 256])
            if args.architecture == '2DA_Dis':
                bin_edges_t, pred_t, pred_domain_label_t1, pred_domain_label_t2 = model(img_t, grl_lambda)
            elif args.architecture == '2Dis':
                bin_edges_t, pred_t, pred_domain_label_t1, pred_domain_label_t2 = model(img_t, grl_lambda)
            else:
                bin_edges_t, pred_t, pred_domain_label_t = model(img_t, grl_lambda)

            # Compute target domain loss
            if args.architecture == 'DA_RWTD':
                # loss_t_domain = Dis(pred_domain_label_t, y_t_domain)
                loss_t_domain = Dis(pred_domain_label_t, y_t_domain, epoch)

            elif args.architecture == 'DA_Dis':
                loss_t_domain = Dis(pred_domain_label_t, y_t_domain)
                # loss_t_domain = Dis(pred_domain_label_t, y_t_domain, epoch)

            elif args.architecture == '2DA_Dis':
                loss_t_domain_1 = NLL_criterion(pred_domain_label_t1, y_t_domain)
                loss_t_domain_2 = Dis(pred_domain_label_t2, y_t_domain)

            elif args.architecture == '2Dis':
                loss_t_domain_1 = Dis1(pred_domain_label_t1, y_t_domain)
                loss_t_domain_2 = Dis(pred_domain_label_t2, y_t_domain)

            elif args.architecture == 'DA':
                loss_t_domain = NLL_criterion(pred_domain_label_t, y_t_domain)

            ###############################################################
            ###                  Train on Source Domain                 ###
            ###############################################################
            y_s_domain = torch.ones(len(img_s), dtype=torch.long).to(device)

            # bin_edges --> [8, 81]
            # pred --> torch.Size([8, 1, 128, 256])
            if args.architecture == '2DA_Dis':
                bin_edges_s, pred_s, pred_domain_label_s1, pred_domain_label_s2 = model(img_s, grl_lambda)
            elif args.architecture == '2Dis':
                bin_edges_s, pred_s, pred_domain_label_s1, pred_domain_label_s2 = model(img_s, grl_lambda)
            else:
                bin_edges_s, pred_s, pred_domain_label_s = model(img_s, grl_lambda)

            # Compute source domain loss
            if args.architecture == 'DA_RWTD':
                # loss_s_domain = Dis(pred_domain_label_s, y_s_domain)
                loss_s_domain = Dis(pred_domain_label_s, y_s_domain, epoch)

            elif args.architecture == 'DA_Dis':
                loss_s_domain = Dis(pred_domain_label_s, y_s_domain)
                # loss_s_domain = Dis(pred_domain_label_s, y_s_domain, epoch)

            elif args.architecture == '2DA_Dis':
                loss_s_domain_1 = NLL_criterion(pred_domain_label_s1, y_s_domain)
                loss_s_domain_2 = Dis(pred_domain_label_s2, y_s_domain)

            elif args.architecture == '2Dis':
                loss_s_domain_1 = Dis1(pred_domain_label_s1, y_s_domain)
                loss_s_domain_2 = Dis(pred_domain_label_s2, y_s_domain)

            elif args.architecture == 'DA':
                loss_s_domain = NLL_criterion(pred_domain_label_s, y_s_domain)

            # mask was --> torch.Size([8, 256, 512, 1])
            mask = depth_s > args.min_depth
            l_dense_s = criterion_ueff(pred_s, depth_s, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer_s = criterion_bins(bin_edges_s, depth_s)
            else:
                l_chamfer_s = torch.Tensor([0]).to(img_s.device)

            ###############################################################
            ###                   Calculate Total Loss                  ###
            ###############################################################
            if args.architecture == 'DA_RWTD' or args.architecture == 'DA_Dis':
                loss = 0.1 * l_dense_s + args.w_chamfer * l_chamfer_s \
                       + 0.001 * (loss_s_domain + loss_t_domain)

                if should_log and step % 5 == 0:
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source'}": loss_s_domain.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target'}": loss_t_domain.item()}, step=step)
                    wandb.log({f"Train/{criterion_ueff.name}": l_dense_s.item()}, step=step)
                    wandb.log({f"Train/{criterion_bins.name}": l_chamfer_s.item()}, step=step)

            elif args.architecture == '2DA_Dis':
                loss = 0.1 * l_dense_s + args.w_chamfer * l_chamfer_s \
                       + 0.001 * (loss_s_domain_2 + loss_t_domain_2) \
                        + 0.5 *(loss_s_domain_1 + loss_t_domain_1)

                if should_log and step % 5 == 0:
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source_First_Classifier_FC'}": loss_s_domain_1.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target_First_Classifier_FC'}": loss_t_domain_1.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source_Second_Classifier_Dis'}": loss_s_domain_2.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target_Second_Classifier_Dis'}": loss_t_domain_2.item()}, step=step)
                    wandb.log({f"Train/{criterion_ueff.name}": l_dense_s.item()}, step=step)
                    wandb.log({f"Train/{criterion_bins.name}": l_chamfer_s.item()}, step=step)

            elif args.architecture == '2Dis':
                loss = 0.1 * l_dense_s + args.w_chamfer * l_chamfer_s \
                       + 0.001 * (loss_s_domain_2 + loss_t_domain_2) \
                        + 0.001 *(loss_s_domain_1 + loss_t_domain_1)

                if should_log and step % 5 == 0:
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source_First_Classifier_Dis'}": loss_s_domain_1.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target_First_Classifier_Dis'}": loss_t_domain_1.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source_Second_Classifier_Dis'}": loss_s_domain_2.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target_Second_Classifier_Dis'}": loss_t_domain_2.item()}, step=step)
                    wandb.log({f"Train/{criterion_ueff.name}": l_dense_s.item()}, step=step)
                    wandb.log({f"Train/{criterion_bins.name}": l_chamfer_s.item()}, step=step)

            elif args.architecture == 'DA':
                loss = 0.1 * l_dense_s + args.w_chamfer * l_chamfer_s \
                       + 0.5 * (loss_s_domain + loss_t_domain)

                if should_log and step % 5 == 0:
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Source'}": loss_s_domain.item()}, step=step)
                    wandb.log({f"Train/{'Domain_Label_Loss_for_Target'}": loss_t_domain.item()}, step=step)
                    wandb.log({f"Train/{criterion_ueff.name}": l_dense_s.item()}, step=step)
                    wandb.log({f"Train/{criterion_bins.name}": l_chamfer_s.item()}, step=step)


            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()

            step += 1

            if args.scheduler:
                scheduler.step()

            ########################################################################################################

            if should_write and step % args.validate_every == 0:
                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, grl_lambda, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)

                model.train()
                #################################################################################################


    if args.architecture == 'DA_RWTD' or args.architecture == 'DA_Dis':
        return model, Dis

    elif args.architecture == '2DA_Dis':
        return model, Dis, Dis1

    elif args.architecture == '2Dis':
        return model, Dis, Dis1

    elif args.architecture == 'DA':
        return model


def validate(args, model, test_loader, grl_lambda, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            # bins, pred = model(img)
            if args.architecture == '2DA_Dis':
                bins, pred, pred_domain_label1, pred_domain_label2 = model(img, grl_lambda)
            elif args.architecture == '2Dis':
                bins, pred, pred_domain_label1, pred_domain_label2 = model(img, grl_lambda)
            else:
                bins, pred, pred_domain_label = model(img, grl_lambda)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            if utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]) is None:
                print()
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=1, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="RWTD")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--root", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--comments", default='', type=str, help="comments for the name of saved model")
    parser.add_argument("--tm", default='../', type=str, help="saved path for trained model")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='3D60', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='', type=str,
                        help="path to dataset")
    # datasets can be found at https://vcl3d.github.io/3D60/
    parser.add_argument('--filenames_file', default='', type=str,
                        help='SUNCG 5p ALL Training dataset (2319) images')
    # unlabelled dataset in experiments can be found at http://3dkim.com/research/VR/EUSIPCO.html
    parser.add_argument('--filenames_file_eval', default='', type=str,
                        help='Stanford2D3D 5p testing dataset (82) images')
    parser.add_argument('--architecture', type=str, help='/DA/DA_Dis/2DA_Dis/2Dis/DA_RWTD', default='DA_RWTD')
    parser.add_argument('--scheduler', type=bool, help='use scheduler or not', default=False)
    parser.add_argument('--input_height', type=int, help='input height', default=256)
    parser.add_argument('--input_width', type=int, help='input width', default=512)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--do_random_rotate', default=False,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')
    parser.add_argument('--data_path_eval',
                        default="/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--logging', type=bool, default=False, help='log with WandB')
    args = parser.parse_args()

    # show the training and testing datasets
    print('The training dataset is ', args.filenames_file.split('/')[-1])
    print('The testing dataset is ', args.filenames_file_eval.split('/')[-1])

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if ngpus_per_node == 1:
        args.gpu = 0
    main_worker(args.gpu, ngpus_per_node, args)
