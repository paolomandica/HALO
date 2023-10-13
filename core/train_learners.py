import os
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, PolynomialLR, ConstantLR
from geoopt.optim import RiemannianSGD, RiemannianAdam

from core.datasets import build_dataset
from core.configs import cfg
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.losses.local_consistent_loss import LocalConsistentLoss
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import load_checkpoint
from core.utils.optim import build_optimizer, build_scheduler
from core.losses.negative_learning_loss import NegativeLearningLoss
from core.active.build import RegionSelection

from core.utils.visualize import visualize_wrong

NUM_WORKERS = 4

np.random.seed(cfg.SEED + 1)
VIZ_LIST = list(np.random.randint(0, 500, 20))


class BaseLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hyper = cfg.MODEL.HYPER
        self.automatic_optimization = False

        # create network
        self.feature_extractor = build_feature_extractor(cfg)

        self.segformer_config = None
        if cfg.MODEL.NAME.startswith("segformer"):
            self.segformer_config = self.feature_extractor.config

        self.classifier = build_classifier(cfg, self.segformer_config)

        # resume checkpoint if needed
        if cfg.resume:
            load_checkpoint(
                self.feature_extractor, cfg.resume, module="feature_extractor"
            )
            load_checkpoint(self.classifier, cfg.resume, module="classifier")

        # create criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # evaluation metrics
        self.intersections = np.array([])
        self.unions = np.array([])
        self.targets = np.array([])

    def forward(self, input_data):
        input_size = input_data.shape[-2:]

        if self.segformer_config is not None:
            return_dict = self.segformer_config.return_dict
            outputs = self.feature_extractor(
                input_data, output_hidden_states=True, return_dict=return_dict
            )
            encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
            out = self.classifier(encoder_hidden_states, size=input_size)
        else:
            out = self.classifier(self.feature_extractor(input_data), size=input_size)
        return out

    def inference(
        self,
        image,
        label,
        flip=True,
        save_embed_path=None,
        save_wrong_path=None,
        cfg=None,
    ):
        size = label.shape[-2:]
        if flip:
            image = torch.cat([image, torch.flip(image, [3])], 0)

        output, embed = self.forward(image)

        if save_wrong_path:
            dir_path = os.path.join(self.cfg.SAVE_DIR, "viz")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            dir_path = os.path.join(self.cfg.SAVE_DIR, "viz", "wrong")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # 640, 1280 --> 1024, 2048  || 160, 320
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=True)
            visualize_wrong(
                image[0], output[:1], embed[:1], label, save_wrong_path, cfg
            )

        if save_embed_path:
            dir_path = os.path.join(self.cfg.SAVE_DIR, "embed")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # embed = F.interpolate(embed, size=size, mode='bilinear', align_corners=True)
            if flip:
                embed = (embed[0] + embed[1].flip(2)) / 2
            else:
                embed = embed[0]
            torch.save(embed.unsqueeze(0).cpu(), save_embed_path)

        output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        return output.unsqueeze(0)

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert output.dim() in [1, 2, 3]
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(
            intersection.float().cpu(), bins=K, min=0, max=K - 1
        )
        area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
        area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return (
            area_intersection.cpu().numpy(),
            area_union.cpu().numpy(),
            area_target.cpu().numpy(),
        )

    def validation_step(self, batch, batch_idx):
        x, y, name = batch["img"], batch["label"], batch["name"]
        name = name[0]
        pred = self.inference(x, y, flip=True)

        output = pred.max(1)[1]
        intersection, union, target = self.intersectionAndUnionGPU(
            output, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL
        )

        intersection = np.expand_dims(intersection, axis=0)
        union = np.expand_dims(union, axis=0)
        target = np.expand_dims(target, axis=0)

        if self.intersections.size == 0:
            self.intersections = intersection
            self.unions = union
            self.targets = target
        else:
            self.intersections = np.concatenate(
                (self.intersections, intersection), axis=0
            )
            self.unions = np.concatenate((self.unions, union), axis=0)
            self.targets = np.concatenate((self.targets, target), axis=0)

    def on_validation_epoch_end(self):
        # gather all the metrics across all the processes
        intersections = self.all_gather(self.intersections)
        unions = self.all_gather(self.unions)
        targets = self.all_gather(self.targets)

        intersections = intersections.flatten(0, 1)
        unions = unions.flatten(0, 1)
        targets = targets.flatten(0, 1)

        # calculate the final mean iou and accuracy
        intersections = intersections.sum(axis=0)
        unions = unions.sum(axis=0)
        targets = targets.sum(axis=0)

        iou_class = intersections / (unions + 1e-10)
        accuracy_class = intersections / (targets + 1e-10)

        mIoU = iou_class.mean() * 100
        mAcc = accuracy_class.mean() * 100
        aAcc = intersections.sum() / (targets.sum() + 1e-10) * 100

        # print metrics table style
        print("\nmIoU: {:.2f}".format(mIoU))
        print("mAcc: {:.2f}".format(mAcc))
        print("aAcc: {:.2f}\n".format(aAcc))

        # log metrics
        self.log(
            "mIoU", mIoU, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "mAcc", mAcc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "aAcc", aAcc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )

        # reset metrics
        self.intersections = np.array([])
        self.unions = np.array([])
        self.targets = np.array([])

    def configure_optimizers(self):
        optim_type = self.cfg.SOLVER.OPTIM
        lr = self.cfg.SOLVER.BASE_LR
        weight_decay = self.cfg.SOLVER.WEIGHT_DECAY
        momentum = self.cfg.SOLVER.MOMENTUM

        # init optimizers
        optimizer_fea = build_optimizer(
            self.feature_extractor,
            self.hyper,
            optim_type,
            lr,
            weight_decay,
            momentum,
        )
        optimizer_cls = build_optimizer(
            self.classifier, self.hyper, optim_type, lr * 10, weight_decay, momentum
        )
        optimizers = [optimizer_fea, optimizer_cls]

        # init schedulers
        scheduler_type = self.cfg.SOLVER.SCHEDULER
        lr_power = self.cfg.SOLVER.LR_POWER
        num_iters = self.cfg.SOLVER.NUM_ITER // len(self.cfg.SOLVER.GPUS)
        warmup_iters = self.cfg.SOLVER.WARMUP_ITERS
        num_iters -= warmup_iters

        scheduler_fea = build_scheduler(
            optimizer_fea,
            scheduler_type,
            num_iters,
            warmup_iters,
            lr_power,
        )
        scheduler_cls = build_scheduler(
            optimizer_cls,
            scheduler_type,
            num_iters,
            warmup_iters,
            lr_power,
        )
        schedulers = [scheduler_fea, scheduler_cls]

        return optimizers, schedulers

    def log_metrics(self, batch_idx):
        self.log("global_step", self.global_step, on_step=True, on_epoch=False)
        base_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("base_lr", base_lr, on_step=True, on_epoch=False)
        if len(self.trainer.optimizers) == 2:
            classifier_lr = self.trainer.optimizers[1].param_groups[0]["lr"]
            self.log("classifier_lr", classifier_lr, on_step=True, on_epoch=False)
        self.log("batch_idx", batch_idx, on_step=True, on_epoch=False)


class SourceLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        src_input, src_label = batch["img"], batch["label"]  # shape [B, 3, 720, 1280]
        src_out = self.forward(src_input)
        src_out = src_out[0]

        loss = self.criterion(src_out, src_label)
        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_metrics(batch_idx)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

        return loss

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode="train", is_source=True)
        self.data_len = len(train_set)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        test_set = build_dataset(self.cfg, mode="test", is_source=True)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return test_loader


class SourceFreeLearner(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.debug = cfg.DEBUG
        if self.debug:
            print(">>>>>>>>>>>>>>>> Debug Mode >>>>>>>>>>>>>>>>")

        # active learning dataloader
        self.active_round = 1
        active_set = build_dataset(
            self.cfg, mode="active", is_source=False, epochwise=True
        )
        self.active_loader = DataLoader(
            dataset=active_set,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # init mask for cityscape
        if (
            "LOCAL_RANK" not in os.environ.keys()
            and "NODE_RANK" not in os.environ.keys()
        ):
            print(">>>>>>>>>>>>>>>> Init Mask >>>>>>>>>>>>>>>>")
            DatasetCatalog.initMask(self.cfg)

        # create criterion
        self.negative_criterion = NegativeLearningLoss()

    def compute_active_iters(self):
        denom = (
            self.cfg.SOLVER.NUM_ITER
            * self.cfg.SOLVER.BATCH_SIZE
            * len(self.cfg.SOLVER.GPUS)
        )
        self.active_iters = [
            int(x * self.data_len / denom) for x in self.cfg.ACTIVE.SELECT_ITER
        ]
        self.print("\nActive learning at iters: {}\n".format(self.active_iters))

    def on_train_start(self):
        self.compute_active_iters()

    def on_train_batch_start(self, batch, batch_idx):
        if self.local_rank == 0 and batch_idx in self.active_iters and not self.debug:
            # save checkpoint
            checkpoint_name = "model_before_round_{}.ckpt".format(self.active_round)
            print("\nSaving checkpoint: {}".format(checkpoint_name))
            self.trainer.save_checkpoint(
                os.path.join(self.cfg.SAVE_DIR, checkpoint_name)
            )

            # start active round
            print(
                f"\n>>>>>>>>>>>>>>>> Active Round {self.active_round} >>>>>>>>>>>>>>>>"
            )
            print(f"batch_idx: {batch_idx}, self.local_rank: {self.local_rank}")

            if self.cfg.ACTIVE.SETTING == "RA":
                RegionSelection(
                    cfg=self.cfg,
                    feature_extractor=self.feature_extractor,
                    classifier=self.classifier,
                    tgt_epoch_loader=self.active_loader,
                    round_number=self.active_round,
                )

            self.log("active_round", self.active_round, on_step=True, on_epoch=False)
            self.active_round += 1
        return batch, batch_idx

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_mask = batch["img"], batch["mask"]
        tgt_out = self.forward(tgt_input)
        if self.hyper:
            tgt_out = tgt_out[0]
        else:
            tgt_out = tgt_out[0]

        predict = torch.softmax(tgt_out, dim=1)
        loss = torch.Tensor([0]).cuda()

        # target active supervision loss
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            loss_sup = self.criterion(tgt_out, tgt_mask)
            loss += loss_sup
            self.log(
                "loss_sup",
                loss_sup.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        # target negative pseudo loss
        if self.cfg.SOLVER.NEGATIVE_LOSS > 0:
            negative_loss = (
                self.negative_criterion(predict) * self.cfg.SOLVER.NEGATIVE_LOSS
            )
            loss += negative_loss
            self.log(
                "negative_loss",
                negative_loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_metrics(batch_idx)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

    def on_train_end(self):
        print("\nSaving last checkpoint...")
        self.trainer.save_checkpoint(os.path.join(self.cfg.SAVE_DIR, "last.ckpt"))

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode="train", is_source=False)
        self.data_len = len(train_set)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg, mode="test", is_source=False)
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader


class SourceTargetLearner(SourceFreeLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        if self.cfg.SOLVER.CONSISTENT_LOSS > 0:
            self.local_consistent_loss = LocalConsistentLoss(
                cfg.MODEL.NUM_CLASSES, cfg.SOLVER.LCR_TYPE
            )

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        # source data
        src_input, src_label = batch[0]["img"], batch[0]["label"]
        src_out = self.forward(src_input)
        if self.hyper:
            src_out = src_out[0]
        else:
            src_out = src_out[0]

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_mask = batch[1]["img"], batch[1]["mask"]
        tgt_out = self.forward(tgt_input)
        if self.hyper:
            tgt_out = tgt_out[0]
        else:
            tgt_out = tgt_out[0]

        predict = torch.softmax(tgt_out, dim=1)
        loss = torch.Tensor([0]).cuda()

        # source supervision loss
        loss_sup = self.criterion(src_out, src_label)
        loss += loss_sup
        self.log(
            "loss_sup",
            loss_sup.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        # target active supervision loss
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            loss_sup_tgt = self.criterion(tgt_out, tgt_mask)
            loss += loss_sup_tgt
            self.log(
                "loss_sup_tgt",
                loss_sup_tgt.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        # source consistency regularization loss
        if self.cfg.SOLVER.CONSISTENT_LOSS > 0:
            consistency_loss = (
                self.local_consistent_loss(src_out, src_label)
                * self.cfg.SOLVER.CONSISTENT_LOSS
            )
            loss += consistency_loss
            self.log(
                "consistency_loss",
                consistency_loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        # target negative pseudo loss
        if self.cfg.SOLVER.NEGATIVE_LOSS > 0:
            negative_loss = (
                self.negative_criterion(predict) * self.cfg.SOLVER.NEGATIVE_LOSS
            )
            loss += negative_loss
            self.log(
                "negative_loss",
                negative_loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_metrics(batch_idx)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

    def train_dataloader(self):
        source_set = build_dataset(self.cfg, mode="train", is_source=True)
        target_set = build_dataset(self.cfg, mode="train", is_source=False)
        self.data_len = len(source_set)
        self.target_len = len(target_set)
        self.print("source data length: ", self.data_len)
        self.print("target data length: ", self.target_len)
        source_loader = DataLoader(
            dataset=source_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        target_loader = DataLoader(
            dataset=target_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return [source_loader, target_loader]


class FullySupervisedLearner(SourceFreeLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        if self.cfg.SOLVER.CONSISTENT_LOSS > 0:
            self.local_consistent_loss = LocalConsistentLoss(
                cfg.MODEL.NUM_CLASSES, cfg.SOLVER.LCR_TYPE
            )

        # remove active learning dataloader
        self.active_loader = None
        self.active_round = 0

    def on_train_start(self):
        return None

    def on_train_batch_start(self, batch, batch_idx):
        return batch, batch_idx

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        # source data
        src_input, src_label = batch[0]["img"], batch[0]["label"]
        src_out = self.forward(src_input)
        if self.hyper:
            src_out = src_out[0]
        else:
            src_out = src_out[0]

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_label = batch[1]["img"], batch[1]["label"]
        tgt_out = self.forward(tgt_input)
        if self.hyper:
            tgt_out = tgt_out[0]
        else:
            tgt_out = tgt_out[0]

        predict = torch.softmax(tgt_out, dim=1)
        loss = torch.Tensor([0]).cuda()

        # source supervision loss
        loss_sup = self.criterion(src_out, src_label)
        loss += loss_sup
        self.log(
            "loss_sup",
            loss_sup.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        # target supervision loss
        loss_sup_tgt = self.criterion(tgt_out, tgt_label)
        loss += loss_sup_tgt
        self.log(
            "loss_sup_tgt",
            loss_sup_tgt.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        # source consistency regularization loss
        if self.cfg.SOLVER.CONSISTENT_LOSS > 0:
            consistency_loss = (
                self.local_consistent_loss(src_out, src_label)
                * self.cfg.SOLVER.CONSISTENT_LOSS
            )
            loss += consistency_loss
            self.log(
                "consistency_loss",
                consistency_loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        # target negative pseudo loss
        if self.cfg.SOLVER.NEGATIVE_LOSS > 0:
            negative_loss = (
                self.negative_criterion(predict) * self.cfg.SOLVER.NEGATIVE_LOSS
            )
            loss += negative_loss
            self.log(
                "negative_loss",
                negative_loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_metrics(batch_idx)

        # manual backward pass
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

    def train_dataloader(self):
        source_set = build_dataset(self.cfg, mode="train", is_source=True)
        target_set = build_dataset(self.cfg, mode="train", is_source=False)
        self.data_len = len(source_set)
        self.target_len = len(target_set)
        self.print("source data length: ", self.data_len)
        self.print("target data length: ", self.target_len)
        source_loader = DataLoader(
            dataset=source_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        target_loader = DataLoader(
            dataset=target_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return [source_loader, target_loader]


class Test(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)

        # evaluation metrics
        self.intersections = np.array([])
        self.unions = np.array([])
        self.targets = np.array([])

    def test_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]

        if "name" in batch.keys():
            name = batch["name"]
            name = name[0]
            name = name.rsplit("/", 1)[-1].rsplit("_", 1)[0]
        else:
            name = str(batch_idx)

        embed_file_name = None
        if self.cfg.TEST.SAVE_EMBED:
            self.save_embeddings(y, name, "label")
            embed_file_name = os.path.join(self.cfg.SAVE_DIR, "embed", name + ".pt")

        wrong_file_name = None
        if self.cfg.TEST.VIZ_WRONG and (batch_idx in VIZ_LIST):
            wrong_file_name = os.path.join(
                self.cfg.SAVE_DIR, "viz", "wrong", name + ".png"
            )

        output = self.inference(
            x,
            y,
            flip=True,
            save_embed_path=embed_file_name,
            save_wrong_path=wrong_file_name,
            cfg=self.cfg,
        )
        pred = output.max(1)[1]

        if self.cfg.TEST.SAVE_EMBED:
            self.save_embeddings(pred, name, "pred")
            self.save_embeddings(output, name, "output")

        intersection, union, target = self.intersectionAndUnionGPU(
            pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL
        )

        intersection = np.expand_dims(intersection, axis=0)
        union = np.expand_dims(union, axis=0)
        target = np.expand_dims(target, axis=0)

        if self.intersections.size == 0:
            self.intersections = intersection
            self.unions = union
            self.targets = target
        else:
            self.intersections = np.concatenate(
                (self.intersections, intersection), axis=0
            )
            self.unions = np.concatenate((self.unions, union), axis=0)
            self.targets = np.concatenate((self.targets, target), axis=0)

    def on_test_epoch_end(self):
        # gather all the metrics across all the processes
        intersections = self.all_gather(self.intersections)
        unions = self.all_gather(self.unions)
        targets = self.all_gather(self.targets)

        intersections = intersections.flatten(0, 1)
        unions = unions.flatten(0, 1)
        targets = targets.flatten(0, 1)

        # calculate the final mean iou and accuracy
        intersections = self.intersections.sum(axis=0)
        unions = self.unions.sum(axis=0)
        targets = self.targets.sum(axis=0)

        iou_class = intersections / (unions + 1e-10)
        accuracy_class = intersections / (targets + 1e-10)

        mIoU = round(iou_class.mean() * 100, 1)
        mAcc = round(accuracy_class.mean() * 100, 1)
        aAcc = round(intersections.sum() / (targets.sum() + 1e-10) * 100, 1)

        # print IoU per class
        print("\n\n")
        print("{:<20}  {:<20}  {:<20}".format("Class", "IoU (%)", "Accuracy (%)"))
        for i in range(cfg.MODEL.NUM_CLASSES):
            print(
                "{:<20}  {:<20.2f}  {:<20.2f}".format(
                    self.class_names[i], iou_class[i] * 100, accuracy_class[i] * 100
                )
            )

        # print mIoUs in LateX format
        print()
        print("mIoU in LateX format:")
        delimiter = " & "
        latex_iou_class = delimiter.join(
            map(lambda x: "{:.1f}".format(x * 100), iou_class)
        )
        print(latex_iou_class + " & " + "{:.1f}".format(mIoU))

        # print metrics table style
        # print()
        # print('mIoU:\t {:.1f}'.format(mIoU))
        # print('mAcc:\t {:.1f}'.format(mAcc))
        # print('aAcc:\t {:.1f}'.format(aAcc))

        # compute mIoU* for synthia-to-cityscapes
        if cfg.MODEL.NUM_CLASSES == 16:
            mIoU_star = 0.0
            for i in range(16):
                if i not in [3, 4, 5]:
                    mIoU_star += iou_class[i]
            mIoU_star = round(mIoU_star / 13 * 100, 1)
            # print('mIoU*:\t {:.1f}\n'.format(mIoU_star))
            self.log(
                "mIoU*",
                mIoU_star,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                prog_bar=True,
            )
        print()

        # log metrics
        self.log(
            "mIoU", mIoU, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True
        )
        self.log(
            "mAcc", mAcc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True
        )
        self.log(
            "aAcc", aAcc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True
        )

    def test_dataloader(self):
        test_set = build_dataset(self.cfg, mode="test", is_source=False)
        self.class_names = test_set.trainid2name
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader

    def save_embeddings(self, output, name, type="embed"):
        dir_path = os.path.join(self.cfg.SAVE_DIR, type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = os.path.join(dir_path, name + ".pt")
        torch.save(output.cpu(), file_name)
