import logging
import os
import pytorch_lightning as pl
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from geoopt.optim import RiemannianSGD

from core.datasets import build_dataset
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import AverageMeter, load_checkpoint, load_checkpoint_ripu
from core.loss.negative_learning_loss import NegativeLearningLoss
from core.active.build import PixelSelection, RegionSelection


class BaseLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hyper = cfg.MODEL.HYPER
        self.automatic_optimization = False

        # create network
        self.feature_extractor = build_feature_extractor(cfg)
        self.classifier = build_classifier(cfg)
        print(self.classifier)

        # resume checkpoint if needed
        if cfg.resume:
            print("Loading checkpoint from {}".format(cfg.resume))
            # load_checkpoint(self.feature_extractor, cfg.resume, module='feature_extractor')
            # load_checkpoint(self.classifier, cfg.resume, module='classifier')
            load_checkpoint_ripu(self.feature_extractor, cfg.resume, module='feature_extractor')
            load_checkpoint_ripu(self.classifier, cfg.resume, module='classifier')

        # create criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # evaluation metrics
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()

    def forward(self, input_data):
        input_size = input_data.shape[-2:]
        out = self.classifier(self.feature_extractor(input_data), size=input_size)
        return out

    def inference(self, image, label, flip=False):
        size = label.shape[-2:]
        if flip:
            image = torch.cat([image, torch.flip(image, [3])], 0)
        output = self.classifier(self.feature_extractor(image))
        if self.hyper:
            output = output[0]
        output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        return output.unsqueeze(0)

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
        area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
        area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection.cpu().numpy(), area_union.cpu().numpy(), area_target.cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch['img'], batch['label'], batch['name']
        pred = self.inference(x, y, flip=True)

        output = torch.argmax(pred, dim=1)
        intersection, union, target = self.intersectionAndUnionGPU(
            output, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)
        self.intersection_meter.update(intersection), self.union_meter.update(union), self.target_meter.update(target)

    def on_validation_epoch_end(self):
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)

        mIoU = iou_class.mean() * 100
        mAcc = accuracy_class.mean() * 100
        aAcc = self.intersection_meter.sum.sum() / (self.target_meter.sum.sum() + 1e-10) * 100

        # print metrics table style
        print('\nmIoU: {:.2f}'.format(mIoU))
        print('mAcc: {:.2f}'.format(mAcc))
        print('aAcc: {:.2f}\n'.format(aAcc))

        self.log('mIoU', mIoU, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('mAcc', mAcc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('aAcc', aAcc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        if self.hyper:
            optim = RiemannianSGD
        else:
            optim = torch.optim.SGD
        optimizer_fea = optim(self.feature_extractor.parameters(), lr=self.cfg.SOLVER.BASE_LR,
                              momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        optimizer_cls = optim(self.classifier.parameters(), lr=self.cfg.SOLVER.BASE_LR * 10,
                              momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        scheduler_fea = torch.optim.lr_scheduler.PolynomialLR(
            optimizer_fea, self.cfg.SOLVER.EPOCHS, power=self.cfg.SOLVER.LR_POWER)
        scheduler_cls = torch.optim.lr_scheduler.PolynomialLR(
            optimizer_cls, self.cfg.SOLVER.EPOCHS, power=self.cfg.SOLVER.LR_POWER)
        return [optimizer_fea, optimizer_cls], [scheduler_fea, scheduler_cls]


class TrainLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        src_input, src_label = batch['img'], batch['label']
        src_out = self.forward(src_input)
        if self.hyper:
            src_out = src_out[0]

        loss = self.criterion(src_out, src_label)
        self.log('loss', loss.item(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

        return loss

    def configure_optimizers(self):
        if self.hyper:
            optim = RiemannianSGD
        else:
            optim = torch.optim.SGD
        optimizer = optim(self.parameters(), lr=self.cfg.SOLVER.BASE_LR,
                          momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, self.cfg.SOLVER.MAX_ITER, power=self.cfg.SOLVER.LR_POWER)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode='train', is_source=True)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader

    def val_dataloader(self):
        test_set = build_dataset(self.cfg, mode='test', is_source=True)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,)
        return test_loader


class ActiveLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.debug = False

        # active learning dataloader
        active_set = build_dataset(self.cfg, mode='active', is_source=False, epochwise=True)
        self.active_loader = DataLoader(
            dataset=active_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,)

        # init mask for cityscape
        if 'LOCAL_RANK' not in os.environ.keys() and 'NODE_RANK' not in os.environ.keys() and not self.debug:
            print(">>>>>>>>>>>>>>>> Init Mask >>>>>>>>>>>>>>>>")
            DatasetCatalog.initMask(self.cfg)

        # create criterion
        self.negative_criterion = NegativeLearningLoss()

        self.active_round = 1

    def on_train_batch_start(self, batch, batch_idx):
        # if self.local_rank == 0 and self.global_step in self.cfg.ACTIVE.SELECT_ITER:
        if self.local_rank == 0 and self.current_epoch in self.cfg.ACTIVE.SELECT_EPOCH and not self.debug:
            print(">>>>>>>>>>>>>>>> Active Round {} >>>>>>>>>>>>>>>>".format(self.active_round))
            if self.cfg.ACTIVE.SETTING == "RA":
                RegionSelection(cfg=self.cfg,
                                feature_extractor=self.feature_extractor,
                                classifier=self.classifier,
                                tgt_epoch_loader=self.active_loader)
            elif self.cfg.ACTIVE.SETTING == 'PA':
                PixelSelection(cfg=self.cfg,
                               feature_extractor=self.feature_extractor,
                               classifier=self.classifier,
                               tgt_epoch_loader=self.active_loader)
        return batch, batch_idx

    def training_step(self, batch, batch_idx):

        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_mask = batch['img'], batch['mask']
        tgt_out = self.forward(tgt_input)
        if self.hyper:
            tgt_out = tgt_out[0]

        predict = torch.softmax(tgt_out, dim=1)
        loss = torch.Tensor([0]).cuda()

        # target active supervision loss
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            loss_sup = self.criterion(tgt_out, tgt_mask)
            loss += loss_sup
            self.log('loss_sup', loss_sup.item(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        # target negative pseudo loss
        if self.cfg.SOLVER.NEGATIVE_LOSS > 0:
            negative_loss = self.negative_criterion(predict) * self.cfg.SOLVER.NEGATIVE_LOSS
            loss += negative_loss
            self.log('negative_loss', negative_loss.item(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        self.log('loss', loss.item(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode='train', is_source=False, epochwise=True)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader

    def val_dataloader(self):
        test_set = build_dataset(self.cfg, mode='test', is_source=False, epochwise=True)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,)
        return test_loader
