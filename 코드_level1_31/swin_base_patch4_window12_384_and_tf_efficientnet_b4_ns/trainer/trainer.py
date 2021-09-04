from cProfile import label
import torch
import time
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class Trainer():
    def __init__(self, model, loss_fn, optimizer, device, scaler=None, logger=None, scheduler=None, schd_batch_update=False) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.schd_batch_update = schd_batch_update
        self.scaler = scaler

    def train_one_epoch(self, epoch, train_loader, cutmix_beta=0, accum_iter=1, verbose_step=1):
        self.model.train()
        matches = 0
        t = time.time()
        running_loss = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(self.device).float()
            image_labels = image_labels.to(self.device).long()

            cutmix = False
            # cutmix 실행 될 경우
            if np.random.random() > 0.5 and cutmix_beta > 0:
                lam = np.random.beta(cutmix_beta, cutmix_beta)
                rand_index = torch.randperm(imgs.size()[0]).to(self.device)
                target_a = image_labels
                target_b = image_labels[rand_index]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index,
                                                        :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2-bbx1)*(bby2-bby1) /
                           (imgs.size()[-1]*imgs.size()[-2]))
                cutmix = True

            with autocast():
                image_preds = self.model(imgs)

                if cutmix:
                    loss = self.loss_fn(image_preds, target_a)*lam + \
                        self.loss_fn(image_preds, target_b)*(1. - lam)
                    cutmix = False
                else:
                    loss = self.loss_fn(image_preds, image_labels)

                self.scaler.scale(loss).backward()

                running_loss += loss.item()
                matches += (np.argmax(image_preds.detach().cpu().numpy(), 1) ==
                            image_labels.detach().cpu().numpy()).sum().item()

                if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
                    # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and self.schd_batch_update:
                        self.scheduler.step()

                if ((step + 1) % verbose_step == 0) or ((step + 1) == len(train_loader)):
                    train_loss = running_loss / verbose_step
                    train_acc = matches / image_labels.shape[0] / verbose_step
                    current_lr = self.get_lr(self.optimizer)
                    description = f'epoch {epoch} loss: {train_loss:.4f} lr: {current_lr} train acc : {train_acc:.2f}'
                    pbar.set_description(description)
                    if self.logger is not None:
                        self.logger.add_scalar("Train/loss", running_loss,
                                               epoch * len(train_loader) + step)
                        self.logger.add_scalar("Train/accuracy", train_acc,
                                               epoch * len(train_loader) + step)
                        self.logger.add_scalar("Train/lr", current_lr,
                                               epoch * len(train_loader) + step)
                    running_loss = 0
                    matches = 0

        img_grid = torchvision.utils.make_grid(imgs)
        self.logger.add_image(f'{epoch}_train_input_img',
                              img_grid, epoch)

        if self.scheduler is not None and not self.schd_batch_update:
            self.scheduler.step()

    # size : [Batch_size, Channel, Width, Height]
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # 패치의 중앙 좌표 값 cx, cy
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 패치 모서리 좌표 값
        bbx1 = 0
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = W
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def valid_one_epoch(self, epoch, valid_loader, verbose_step=1, summary=True, return_f1=True, schd_loss_update=False):
        self.model.eval()

        loss_sum = 0
        sample_num = 0
        image_preds_all = []
        image_targets_all = []

        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(self.device).float()
            image_labels = image_labels.to(self.device).long()

            image_preds = self.model(imgs)  # output = model(input)
            image_preds_all += [torch.argmax(image_preds,
                                             1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]

            loss = self.loss_fn(image_preds, image_labels)
            loss_sum += loss.item()*image_labels.shape[0]
            sample_num += image_labels.shape[0]

            if ((step + 1) % verbose_step == 0) or ((step + 1) == len(valid_loader)):
                description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                pbar.set_description(description)
                valid_acc = (np.concatenate(image_preds_all) == np.concatenate(
                    image_targets_all)).sum().item() / image_labels.shape[0] / (step + 1)
                if self.logger is not None:
                    self.logger.add_scalar("Valid/accuracy", valid_acc,
                                           epoch * len(valid_loader) + step)
                    self.logger.add_scalar("Valid/loss", loss_sum/sample_num,
                                           epoch * len(valid_loader) + step)

        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)

        print('validation multi-class f1_score = {:.4f}'.format(
            f1_score(image_preds_all, image_targets_all, average='macro')))
        if summary:
            print(f'{epoch} epoch report')
            print(classification_report(image_targets_all, image_preds_all))

        if self.scheduler is not None:
            if schd_loss_update:
                self.scheduler.step(loss_sum/sample_num)
            else:
                self.scheduler.step()
        if return_f1:
            return f1_score(image_preds_all, image_targets_all, average='macro')
