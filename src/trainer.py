from datetime import datetime
import logging
import math
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer

from src import logger_utils, utils
from src.entities import Dataset
# from src.evaluation import Evaluator
# from src.greedy_decode import GreedyDecoder
from src.model import Seq2SeqTransformer
from src.noam_scheduler import NoamScheduler
from src.loss import Loss
from src.reader import Reader


class Trainer:
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self._device = torch.device(utils.get_device())
        self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if utils.get_device() == 'cuda':
            self._logger.info(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        utils.set_seed(args.seed)

    def train(self):
        args = self.args
        self._log_arguments()
        self._logger.info(f'Model: {args.model_name}')

        # Read data
        reader = Reader(self._tokenizer)
        train_dataset = reader.read(src_path=args.src_train_data, tgt_path=args.tgt_train_data,
                                    data_name=args.data_name, max_length=args.max_length)
        valid_dataset = reader.read(src_path=args.src_valid_data, tgt_path=args.tgt_valid_data,
                                    data_name=args.data_name, max_length=args.max_length)
        self._log_dataset(train_dataset, valid_dataset)
        n_train_samples = len(train_dataset)
        updates_epoch = n_train_samples // args.train_batch_size
        n_updates = updates_epoch * args.num_epochs

        # Start training
        self._logger.info('--------------  Running training  --------------')
        self._logger.info(f'Updates per epoch: {updates_epoch}')
        self._logger.info(f'Updates total: {n_updates}')

        # Create model and optimization
        model = Seq2SeqTransformer(n_encoder_layers=args.num_encoder_layers, n_decoder_layers=args.num_decoder_layers,
                                   emb_size=args.embedding_size, n_heads=args.num_heads,
                                   src_vocab_size=len(self._tokenizer.get_vocab()),
                                   tgt_vocab_size=len(self._tokenizer.get_vocab()),
                                   dim_feedforward=args.ffn_hidden_dim, dropout=args.transformers_dropout)
        model.to(self._device)
        optimizer_params = model.parameters()
        optimizer = AdamW(optimizer_params, lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=round(args.lr_warmup * n_updates),
                                                                 num_training_steps=n_updates)
        # scheduler = NoamScheduler(optimizer, args.ffn_hidden_dim, 4000)

        # Create loss function
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self._tokenizer.pad_token_id)
        loss_calculator = Loss(criterion=criterion, model=model, optimizer=optimizer, scheduler=scheduler,
                               max_grad_norm=args.max_grad_norm)

        # Start training
        best_valid_loss = float('inf')
        global_loss = 0.0
        global_iteration = 0
        best_time = 0
        for epoch in range(args.num_epochs):
            self._logger.info(f'-----------  EPOCH {epoch}  -----------')
            train_dataset.set_mode(Dataset.TRAIN_MODE)
            data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                     collate_fn=self._collate_fn)
            model.zero_grad()
            for batch in tqdm(data_loader, total=updates_epoch - 1, desc=f'Train epoch {epoch}'):
                batch_loss = self._train_iter(model, batch, loss_calculator)
                global_loss += batch_loss

                if (global_iteration + 1) % self.args.log_iter == 0:
                    log_loss = global_loss / self.args.log_iter
                    self._log_train(scheduler, log_loss, epoch, global_iteration)
                    global_loss = 0.0

                if self.args.steps_per_eval is not None and (global_iteration + 1) % self.args.steps_per_eval == 0:
                    valid_loss = self._compute_loss_eval(model, valid_dataset, loss_calculator, global_iteration)
                    self._log_valid(valid_loss, global_iteration)
                    if valid_loss < best_valid_loss:
                        self._logger.info(f'New best loss, from {best_valid_loss} to {valid_loss}, '
                                          f'at global iteration {global_iteration}')
                        old_best_model = os.path.join(self._path, f'bestLossModel_iter={best_time}.pt')
                        if os.path.exists(old_best_model):
                            os.remove(old_best_model)
                        best_valid_loss = valid_loss
                        best_time = global_iteration
                        self._save_model(model, optimizer, scheduler, epoch,
                                         flag=f'bestLossModel_iter={global_iteration}')

                    # decoder = GreedyDecoder(model, self._tokenizer, self._device, self.args.max_length)
                    # evaluator = Evaluator(valid_dataset, self._tokenizer, decoder)
                    # print(evaluator.compute_scores())
                global_iteration += 1

            if self.args.steps_per_eval is None:
                valid_loss = self._compute_loss_eval(model, valid_dataset, loss_calculator, global_iteration)
                self._log_valid(valid_loss, global_iteration)
                if valid_loss < best_valid_loss:
                    self._logger.info(f'New best loss, from {best_valid_loss} to {valid_loss}, at epoch {epoch}')
                    old_best_model = os.path.join(self._path, f'bestLossModel_epoch={best_time}.pt')
                    if os.path.exists(old_best_model):
                        os.remove(old_best_model)
                    best_valid_loss = valid_loss
                    best_time = epoch
                    self._save_model(model, optimizer, scheduler, epoch, flag=f'bestLossModel_epoch={epoch}')

        # Save final model
        self._save_model(model, optimizer, scheduler, args.num_epochs - 1,
                         flag=f'finalModel_epoch={args.num_epochs - 1}')
        self._logger.info('------------- Finish training!!! -------------')

    def _train_iter(self, model, batch, loss_calculator):
        model.train()
        batch = utils.to_device(batch, self._device)
        src = batch['encoding']
        tgt = batch['label']
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(src, tgt_input, model)
        logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        batch_loss = loss_calculator.compute(logits, tgt_expected, mode='train')

        return batch_loss

    def _compute_loss_eval(self, model, dataset, loss_calculator, time):
        if self.args.steps_per_eval is not None:
            desc = f'Compute loss in validation dataset at global iteration {time}'
        else:
            desc = f'Compute loss in validation dataset at epoch {time}'
        dataset.set_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                 collate_fn=self._collate_fn)
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            total = math.ceil(len(dataset) / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc=desc):
                batch = utils.to_device(batch, self._device)
                src = batch['encoding']
                tgt = batch['label']
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(src, tgt_input, model)
                logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                               src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
                batch_loss = loss_calculator.compute(logits, tgt_expected, mode='eval')
                batch_size = src.shape[0]
                total_loss += batch_loss * batch_size
        loss = total_loss / len(dataset)

        return loss

    def _create_mask(self, src, tgt, model: Seq2SeqTransformer):
        """
        Create mask
        :param src: shape [batch_size, src_seq_length]
        :param tgt: shape [batch_size, tgt_seq_length]
        :param model:
        :return:
        """
        src_seq_length = src.shape[1]
        tgt_seq_length = tgt.shape[1]

        src_mask = torch.zeros((src_seq_length, src_seq_length), device=self._device).type(torch.bool)
        tgt_mask = model.generate_square_subsequent_mask(sz=tgt_seq_length).to(self._device)

        src_padding_mask = (src == self._tokenizer.pad_token_id).to(self._device)
        tgt_padding_mask = (tgt == self._tokenizer.pad_token_id).to(self._device)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()

        for key in keys:
            samples = [s[key] for s in batch]
            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                if key == 'encoding' or key == 'label':
                    padding = self._tokenizer.pad_token_id
                else:
                    padding = 0
                padded_batch[key] = utils.padded_stack(samples, padding=padding)

        return padded_batch

    def _init_logger(self):
        time = str(datetime.now()).replace(' ', '_').replace(':', '-')[:-7]
        log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s [%(levelname)-5.5s] %(message)s')
        self._logger = logging.getLogger()
        logger_utils.reset_logger(self._logger)

        if self.args.mode == 'train':
            self._path = os.path.join(self.args.train_path, time)
            self._log_path = os.path.join(self._path, 'log')
            os.makedirs(self._path, exist_ok=True)
            os.makedirs(self._log_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
            self._loss_csv = os.path.join(self._path, 'loss.csv')
            logger_utils.create_csv(self._loss_csv,
                                    header=['global_iteration', 'epoch', 'train_loss', 'current_lr'])
            self._eval_csv = os.path.join(self._path, 'eval.csv')
            if self.args.steps_per_eval is not None:
                logger_utils.create_csv(self._eval_csv, header=['steps', 'valid_loss'])
            else:
                logger_utils.create_csv(self._eval_csv, header=['epoch', 'valid_loss'])
            os.makedirs(os.path.join(self._path, self.args.tensorboard_path), exist_ok=True)
            self._writer = SummaryWriter(os.path.join(self._path, self.args.tensorboard_path))
        else:
            self._eval_path = os.path.join(self.args.eval_path, time)
            os.makedirs(self._eval_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._eval_path, 'all.log'))

        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging.INFO)

    def _log_arguments(self):
        logger_utils.log_json(path=self._path, data=self.args, name='args')

    def _log_dataset(self, train_dataset: Dataset, valid_dataset: Dataset):
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Train dataset: {len(train_dataset)} samples')
        self._logger.info(f'Validation dataset: {len(valid_dataset)} samples')

    def _log_train(self, scheduler, loss, epoch, global_iteration):
        lr = scheduler.get_last_lr()
        data = [global_iteration, epoch, loss, lr]
        logger_utils.log_csv(self._loss_csv, data)
        self._logger.info(f'Training loss at epoch {epoch}, global iteration {global_iteration}: {loss}')
        self._writer.add_scalar('Training loss per iteration', loss, global_step=global_iteration)
        self._writer.add_scalar('Learning rate per iteration', lr[0], global_step=global_iteration)

    def _log_valid(self, loss, time):
        if self.args.steps_per_eval is not None:
            self._logger.info(f'Validation loss at global iteration {time}: {loss}')
            self._writer.add_scalar('Validation loss per iteration', loss, global_step=time)
        else:
            self._logger.info(f'Validation loss at epoch {time}: {loss}')
            self._writer.add_scalar('Validation loss per epoch', loss, global_step=time)
        logger_utils.log_csv(self._eval_csv, [time, loss])

    def _save_model(self, model: nn.Module, optimizer, scheduler, epoch: int, flag: str):
        save_path = os.path.join(self._path, flag + '.pt')
        saved_point = {'model': model,
                       'optimizer': optimizer,
                       'scheduler': scheduler.state_dict(),
                       'epoch': epoch}
        torch.save(saved_point, save_path)

    def _load_model(self, model_path: str):
        saved_point = torch.load(model_path, map_location=self._device)

        return saved_point['model']
