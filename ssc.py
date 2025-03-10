import sys
import uuid
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F, PairwiseDistance
from tqdm import trange

import utils
from ssc_common import proto, get_args, ModelWrapper,  wrapped_model_test_extracted,\
    _test_extracted, Transformation, load_train_test_data, load_wrapped_model
from utils import set_seed

def estimate_u(Y, F, outliers, n_classes, logger):
    npps_in, nms_in = proto(Y, F, (~outliers), n_classes)
    npps_out, nms_out = proto(Y, F, (outliers), n_classes)
    logger.write(nms_out)
    C = (0.5 * (npps_in + npps_out))[:-1]
    mask = (nms_out[:-1] > 1).float()
    C = mask[:, None] * C + (1 - mask)[:, None] * npps_in[:-1]
    return C

def get_centers(data, n_classes, logger, outliers=None):
    Y, F = data['y'].flatten(), data['f']
    C_global = None
    if outliers is None:
        C_global = proto(Y, F, mask=None, n_classes=n_classes)
        outliers = (F[:, None] - C_global[0][None, :-1]).norm(dim=-1, p=2).argmin(dim=-1) != Y
    C = estimate_u(Y, F, outliers, n_classes, logger)
    return C, C_global[0][:-1], outliers

def transform_feature(model, centers, data, N, lr=1e-3):
    feat = data['f'].detach().clone()
    Y = data['y'].flatten().long().detach().clone()
    beta = nn.Parameter(torch.ones((1,feat.size(-1)), requires_grad=True).detach().clone().float().cuda())
    bias = nn.Parameter(torch.zeros((1,feat.size(-1)), requires_grad=True).detach().clone().float().cuda())
    optimizer = torch.optim.AdamW([beta, bias], lr, weight_decay=0)
    transformation = Transformation(beta,bias)#lambda x: (x + bias) * beta- bias
    dist = PairwiseDistance(keepdim=True)
    for i in trange(N):
        idx = torch.arange(0, len(feat))
        optimizer.zero_grad()
        f, p, y = transformation(feat[idx]), centers[Y[idx]], Y[idx]
        loss = dist(f, p).mean() + 10 * beta.mean()
        loss.backward()
        optimizer.step()
    return transformation


def adjust_classifier(fc, data, N=100, mask=None, B=64, mask2=None, criterion=F.cross_entropy):
    group1 = {k: v[mask] for k, v in data.items()}
    mask2 = (~mask) if mask2 is None else mask2
    group2 = {k: v[mask2] for k, v in data.items()}
    feat1, feat2 = group1['f'].detach().clone(), group2['f'].detach().clone()
    Y1, Y2 = group1['y'].flatten().long().detach().clone(), group2['y'].flatten().long().detach().clone()
    fc_new = nn.Linear(fc.in_features, fc.out_features).cuda()
    optimizer = torch.optim.AdamW(fc_new.parameters(), 1e-3,weight_decay=0)
    B = min(len(Y1), len(Y2))
    for i in trange(N):
        optimizer.zero_grad()
        idx1 = torch.randint(0, len(Y1), (B,)) if B < len(Y1) else torch.arange(0, len(Y1))
        idx2 = torch.randint(0, len(Y2), (B,)) if B < len(Y2) else torch.arange(0, len(Y2))
        loss = criterion(fc_new(feat1[idx1]), Y1[idx1]) + criterion(fc_new(feat2[idx2]), Y2[idx2])
        loss.backward()
        optimizer.step()
    return fc_new

# wx+b
def main_exp(all_train_data, all_test_data, n_classes, logger, N1, N2, model=None):
    n_classes=int(n_classes)
    train_data, test_data = all_train_data, all_test_data
    plog=lambda title,x:print(f"{title} mAcc={x[1]}, wga={x[0]}")
    plog("ERM,x",wrapped_model_test_extracted(model, all_test_data, logger=None, B=4096))
    centers, C_global, outliers = get_centers(train_data, n_classes, logger)
    transformation = transform_feature(model, centers, train_data, N1)
    test_data = {k: v for k, v in test_data.items()}
    test_data['f'] = transformation(test_data['f']).detach()
    train_data = {k: v for k, v in train_data.items()}
    train_data['f'] = transformation(train_data['f']).detach()

    Y,PRED_ORI= _test_extracted(C_global, all_train_data, logger, B=4096)
    Y,PRED_NEW= _test_extracted(centers,train_data,logger,B=4096)
    mask=torch.logical_and(PRED_ORI!=Y,PRED_NEW==Y).flatten().cuda()
    mask2=(PRED_ORI==Y).flatten().cuda()
    fc_new=adjust_classifier(model.fc, train_data, N2, mask, mask2=mask2)
    model2=ModelWrapper(model,fc_new)

    plog("debiased,x",wrapped_model_test_extracted(model2, all_test_data, logger=None, B=4096))
    return (wrapped_model_test_extracted(model2, test_data, logger, B=4096))

def run_exp(args,logger):
    shared_vars = dict(train_split='val', test_split='test', args=args, logger=logger)
    logger.write(args)
    logger.write(f"{args.data_dir}")
    all_train_data, all_test_data=load_train_test_data(args,shared_vars)
    n_classes = all_train_data['y'].max() + 1
    model= load_wrapped_model(shared_vars)
    logger.write("wx+b")
    return main_exp(all_train_data, all_test_data, n_classes, logger, model=model, N1=args.num_epochs, N2=args.num_epochs_ft)


if __name__ == '__main__':
    all_args_map={
        # "chexpert":['--data_dir=chexpert',
        #           '--data_transform=AugWaterbirdsCelebATransform', '--dataset=SpuriousCorrelationDataset',
        #           '--model=imagenet_resnet50_pretrained', '--ckpt_path=logs/chexpert/erm_seed1/final_checkpoint.pt',
        #           '--label_filename=metadata.csv', '--batch_size=64', '--num_epochs=650', '--num_epochs_ft=10000'],
        "waterbirds":['--data_dir=waterbirds',
                  '--data_transform=AugWaterbirdsCelebATransform', '--dataset=SpuriousCorrelationDataset',
                  '--model=imagenet_resnet50_pretrained', '--ckpt_path=logs/waterbirds/erm_seed1/final_checkpoint.pt',
                  '--label_filename=metadata.csv', '--batch_size=64', '--num_epochs=10', '--num_epochs_ft=500'],
        "celeba":['--data_dir=celeba',
                  '--data_transform=NoAugWaterbirdsCelebATransform', '--dataset=SpuriousCorrelationDataset',
                  '--model=imagenet_resnet50_pretrained', '--ckpt_path=logs/celeba/erm_seed1/final_checkpoint.pt',
                  '--label_filename=metadata.csv', '--batch_size=64', '--num_epochs=350', '--num_epochs_ft=110'],
        "multinli":[
            '--data_dir=multinli',
            '--data_transform=None',
            '--n_classes=3',
            '--dataset=MultiNLIDataset',
            '--model=bert_pretrained',
            '--ckpt_path=logs/multinli/erm_seed1/final_checkpoint.pt', '--num_epochs=1200', '--num_epochs_ft=1500'],
        "civilcomments":['--data_dir=cc',
                  '--data_transform=BertTokenizeTransform', '--dataset=WildsCivilCommentsCoarse',
                  '--model=bert_pretrained', '--batch_size=64',
                  '--ckpt_path=logs/civilcomments/erm_seed1/final_checkpoint.pt', '--num_epochs=300'],
    }

    exp_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
               f'{str(uuid.uuid4())[:5]}'

    datasets=[sys.argv[1]]
    args_map={k:all_args_map[k] for k in datasets} if datasets is not None else all_args_map
    logger = utils.Logger(fpath=f"{exp_name}.log")
    for dataset_name, exp_args in args_map.items():
        args = get_args(exp_args)
        wga, mean, c = 0, 0, 0
        W, M = [], []
        for i in range(1):
            set_seed(i)
            w,m=(run_exp(args,logger))
            print(f"seed {i} {dataset_name} macc:{m}, wga:{w}")
            wga+=w
            mean+=m
            c+=1
            W+=[w]
            M+=[m]
        Mmacc,Mwga=torch.cat([w[None] for w in M]).mean(),torch.cat([w[None] for w in W]).mean()
        STDmacc,STDwga=torch.cat([w[None] for w in M]).std(),torch.cat([w[None] for w in W]).std()
        print(f"Dataset {dataset_name} mean mAcc={Mmacc}+-{STDmacc}, mean wga={Mwga}+-{STDwga}")


