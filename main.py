import os
import argparse
import traceback
import shutil
import pytz
import logging
import wandb
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.general import *
from utils.data import *

from model.model import MIXAD, count_params

from train import train
from test  import test
from evaluate import *


class Main():
    def __init__(self, args):
        self.args = args
        self.args.logger.info('*** ARGUMENT SETTING ***')
        for arg, value in vars(args).items():
            self.args.logger.info(f'Argument {arg} : {value}')
        self.args.logger.info('\n')
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")

        ## Load already normalized data
        dataset = self.args.dataset
        train, val, test, labels = load_data(dataset, val_ratio=args.val_ratio)

        self.args.logger.info('*** DATA CONFIGURATION ***')
        self.args.logger.info(f'Dataset: {dataset}')
        self.args.logger.info(f'Train shape: {train.shape} (normalized, w/o attack labels)')
        self.args.logger.info(f'Validation shape: {val.shape} (normalized, w/o attack labels)')
        self.args.logger.info(f'Test shape: {test.shape} (normalized, w/o attack labels)')
        self.args.logger.info(f'Label shape: {labels.shape} (label for every features & timestamps)')
        self.args.logger.info('\n')

        ## Dataset & Dataloader
        self.train_dataset = TimeDataset(args, train, np.zeros_like(train))
        self.val_dataset = TimeDataset(args, val, np.zeros_like(val))
        self.test_dataset = TimeDataset(args, test, labels)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=1, worker_init_fn=lambda _: np.random.seed(args.seed))
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=1, worker_init_fn=lambda _: np.random.seed(args.seed))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=1, worker_init_fn=lambda _: np.random.seed(args.seed))

        ## Define model
        self.model = MIXAD(self.args).to(self.device)
        self.args.logger.info('*** MODEL CONFIGURATION ***')
        self.args.logger.info(self.model)
        self.args.logger.info('\n')
        self.args.logger.info(f"Total trainable parameters {count_params(self.model)}")
        self.args.logger.info('\n')


    def run(self):
        if self.args.load_dir == '':
            train(args = self.args,
                  model = self.model,
                  train_dataloader = self.train_dataloader,
                  val_dataloader = self.val_dataloader,
                  test_dataloader = self.test_dataloader)
            
            self.model.load_state_dict(torch.load(f'{self.args.log_dir}/best.pth'))
            best_model = self.model.to(self.device)
            self.args.logger.info(f'Best model loaded from : {self.args.log_dir}/best.pth')

        else:
            self.model.load_state_dict(torch.load(f'{self.args.load_dir}/best.pth'))
            best_model = self.model.to(self.device)
            self.args.logger.info(f'Best model loaded from : {self.args.load_dir}/best.pth')
        self.args.logger.info('\n')

        test_results = test(args, best_model, self.test_dataloader)
        info = evaluation(args, test_results)
        ad_results = info[0]
        f1 = ad_results['f1']
        pre = ad_results['precision']
        rec = ad_results['recall']

        self.args.logger.info('*** Detection Result ***')
        self.args.logger.info(f'F1 score: {f1:.4f}')
        self.args.logger.info(f'Precision: {pre:.4f}')
        self.args.logger.info(f'Recall: {rec:.4f}')

        if args.interpretation:
            _, interp_results, (anomaly_scores_feature, labels_feature) = info
            hr100 = interp_results['Hit@100%']
            hr150 = interp_results['Hit@150%']

            self.args.logger.info('*** Interpretation Result ***')
            self.args.logger.info(f'HitRate@100%: {hr100:.4f}')
            self.args.logger.info(f'HitRate@150%: {hr150:.4f}')

            return [pre, rec, f1, hr100, hr150]
        else:
            return [pre, rec, f1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-dataset', help='dataset name', type=str, default='SMD_1_1')
    parser.add_argument('-num_nodes', help='number of nodes', type=int, default=38)
    parser.add_argument('-seq_len', help='input sequence length', type=int, default=30)
    parser.add_argument('-horizon', help='output sequence length', type=int, default=1)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('-input_dim', help='number of input channel', type=int, default=1)
    parser.add_argument('-output_dim', help='number of output channel', type=int, default=1)
    # model
    parser.add_argument('-max_diffusion_step', help='max diffusion step or Cheb K', type=int, default=3)
    parser.add_argument('-num_rnn_layers', help='number of rnn layers', type=int, default=1)
    parser.add_argument('-rnn_units', help='number of rnn units', type=int, default=64)
    parser.add_argument('-mem_num', help='number of meta-nodes/prototypes', type=int, default=5)
    parser.add_argument('-mem_dim', help='dimension of meta-nodes/prototypes', type=int, default=64)
    # train
    parser.add_argument("-loss", help="mask_mae_loss", type=str, default='mask_mae_loss')
    parser.add_argument('-lamb_cont', help='lamb value for contrastive loss', type=float, default=0.01)
    parser.add_argument('-lamb_cons', help='lamb value for consistency loss', type=float, default=0.1)
    parser.add_argument('-lamb_kl', help='lamb value for kl loss', type=float, default=0.0001)
    parser.add_argument("-epochs", help="number of epochs of training", type=int, default=30)
    parser.add_argument("-patience", help="patience used for early stop", type=int, default=10)
    parser.add_argument("-batch_size", help="size of the batches", type=int, default=256)
    parser.add_argument("-lr", help="base learning rate", type=float, default=0.001)
    parser.add_argument("-steps", help="steps", type=eval, default=[50, 100])
    parser.add_argument("-lr_decay_ratio", help="lr_decay_ratio", type=float, default=0.1)
    parser.add_argument("-epsilon", help="optimizer epsilon", type=float, default=1e-3)
    parser.add_argument("-max_grad_norm", help="max_grad_norm", type=int, default=5)
    parser.add_argument("-use_curriculum_learning", help="use_curriculum_learning", type=eval, choices=[True, False], default='True')
    parser.add_argument("-cl_decay_steps", help="cl_decay_steps", type=int, default=2000)
    # test
    parser.add_argument('-n_th_steps', help='number of intervals of threshold to test for', type=int, default=100)
    parser.add_argument('-load_dir', help='pretrained model directory', type=str, default='')
    parser.add_argument('-interpretation', help='evaluate interpretation', type=eval, choices=[True, False], default='True')
    parser.add_argument('-wandb', help='wandb logging', type=eval, choices=[True, False], default='False')
    # log
    parser.add_argument('-log_dir', help='model save directory', type=str, default='')
    parser.add_argument('-comment', help='experiment comment', type=str, default='comment')
    # misc
    parser.add_argument('-seed', help='random seed', type=int, default=42)
    parser.add_argument('-device', help='cuda:0 / cpu', type=str, default='cuda:0')
    
    args = parser.parse_args()
    fix_seed(args.seed)

    local_tz = pytz.timezone('Asia/Seoul')
    local_time = datetime.now(local_tz)

    log_name = local_time.strftime("%m-%d-%HH%MM%Ss")+f"_{args.comment}"
    if args.load_dir == '': 
        args.log_dir = f'./results/{args.dataset}/{log_name}'
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    else: 
        args.log_dir = f'{args.load_dir}/{log_name}'
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    logging.basicConfig(
        filename = f'{args.log_dir}/results.log',
        format = '%(asctime)s ::: %(message)s',
        level = logging.INFO,
        datefmt = '%m/%d/%Y %H:%M:%S'
    )

    args.logger = logging.getLogger('tqdm_logger')
    handler = TqdmLoggingHandler()
    args.logger.addHandler(handler)

    if args.wandb:
        wandb.init(
            project = 'MIXAD',
            config = args,
        )
        wandb.run.name = f'{args.dataset}_{log_name}'

    main = Main(args)
    results_to_save = main.run()
    msg = ''
    for i, val in enumerate(results_to_save):
        if i == len(results_to_save) - 1:
            msg += f'{val:.4f}'
        else:
            msg += f'{val:.4f},'

    target_data = args.dataset.split('_')[0] if '_' in args.dataset else args.dataset
    f = open(f'../results/{target_data}_results_{args.comment}.txt', 'a')     
    f.write(f'{args.dataset}: {msg}\n')
    f.close()

