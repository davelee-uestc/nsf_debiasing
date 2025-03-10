import gc
import os
import utils
import mmcv
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def md5sum(str):
    from hashlib import md5
    md5_hash = md5()
    md5_hash.update(str.encode('utf-8'))
    return md5_hash.hexdigest()


def _load_wrapped_model(args, logger):
    model = load_model(args, logger)
    model.eval()
    model2 = ModelWrapper(model, model.fc)
    model.fc = torch.nn.Identity()
    return model2

def load_wrapped_model(shared_vars):
    if 'model' not in shared_vars:
        shared_vars['model'] = _load_wrapped_model(shared_vars['args'],shared_vars['logger'])
    return shared_vars['model']

def load_datasets(shared_vars):
    if 'loader_dict' not in shared_vars:
        shared_vars['loader_dict'] = _load_datasets(shared_vars['args'],shared_vars['logger'])
    return shared_vars['loader_dict']

def _load_datasets(args, logger):
    train_loader, test_loader_dict, get_ys_func = (
        utils.get_data(args, logger, contrastive=False))

    loader_dict = {k: v for k, v in test_loader_dict.items()}
    loader_dict['train'] = train_loader
    return loader_dict


def load_cached_embeddings(fp, split, fields, shared_vars):
    if os.path.exists(fp):
        all_data = torch.load(fp)
        print(f"load cached embeddings {fp}")
        # args.n_classes = all_data['y'].max() + 1
    else:
        model2, loader_dict = load_wrapped_model(shared_vars),load_datasets(shared_vars)
        all_data = extract_data(model2, loader_dict[split],
                                mask_fn=lambda y, *a, **b: y > -1,
                                fields=fields
                                )
        torch.save(all_data, fp)
    all_data = {k: v.cuda() for k, v in all_data.items()}
    return all_data

def load_train_test_data(args,shared_vars):
    md5_hex = md5sum(args.ckpt_path)
    emb_split_fp = {
        'train': os.path.join(args.data_dir, f'{shared_vars["train_split"]}_{md5_hex}_emb.pth'),
        'test': os.path.join(args.data_dir, f'{shared_vars["test_split"]}_{md5_hex}_emb.pth')
    }
    all_train_data = load_cached_embeddings(emb_split_fp['train'], fields=dict(
        f=lambda cond, f, *a, **b: f[cond].detach().clone(),
        y=lambda cond, y, *a, **b: y[cond][:, None].detach().clone(),
        loss=lambda cond, loss, *a, **b: loss[cond][:, None].detach().clone(),
    ),
                                            split=shared_vars['train_split'],
                                            shared_vars=shared_vars
                                            )
    all_test_data = load_cached_embeddings(emb_split_fp['test'], fields=dict(
        f=lambda cond, f, *a, **b: f[cond].detach().clone(),
        y=lambda cond, y, *a, **b: y[cond][:, None].detach().clone(),
        p=lambda cond, p, *a, **b: p.cuda()[cond][:, None].detach().clone(),
    ),
                                           split=shared_vars['test_split'],
                                           shared_vars=shared_vars
                                           )
    return all_train_data,all_test_data

def proto(labels_s, f_s, mask=None,n_classes=3):
    labels_s = labels_s.clone()
    if mask is not None:
        labels_s[mask <= 0] = n_classes
    label_one_hot = F.one_hot(labels_s, n_classes + 1).permute(1, 0)
    if f_s.size(0)!=label_one_hot.size(1):
        pass
    npps_sum = label_one_hot.float() @ f_s
    label_masks = label_one_hot.sum(dim=-1) + 1e-5
    labelw = 1 / (label_masks.unsqueeze(dim=-1).expand(npps_sum.size()))
    npps = labelw * npps_sum
    label_masks[-1] = 0
    label_masks[label_masks < 1] = 0
    nms = label_masks
    return npps, nms


class ModelWrapper(nn.Module):
    def __init__(self, model, fc):
        super().__init__()
        self.model=model
        self.n_classes=model.n_classes
        self.out_features=model.out_features
        self.fc=fc

    def forward(self,*args,**kwargs):
        x=self.model(*args,**kwargs)
        return x,self.fc(x)

class Transformation(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w=w
        self.b=b

    def forward(self,x):
        return (x + self.b) * self.w- self.b

def get_args(sysargs=None):
    import utils
    parser = utils.get_model_dataset_args()

    parser.add_argument(
        "--ckpt_path", type=str, default=None, required=False,
        help="Checkpoint path")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--num_epochs_ft", type=int, default=100)
    parser.add_argument(
        "--n_classes", type=int, default=2, required=False,
        help="n_classes")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--log_dir", type=str, default="", help="For loading wandb results")
    parser.add_argument(
        "--label_filename", type=str, default='metadata_random.csv', required=False,
        help="label filename")

    args = parser.parse_args(args=sysargs)
    args.num_minority_groups_remove = 0
    args.reweight_groups = False
    args.reweight_spurious = False
    args.reweight_classes = False
    args.no_shuffle_train = False
    args.shuffle_val=False
    args.mixup = False
    args.load_from_checkpoint = True
    return args

def simple_draw(img,title=None,id=None):
    plt.imshow(img)
    if title is not None:
        plt.title(title if id is None else '{}_{}'.format(id,title))
    plt.show()

def load_model(args,logger):
    n_classes=args.n_classes
    import models
    # Model
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    model.n_classes = n_classes
    model.out_features = model.fc.in_features
    if args.ckpt_path and args.load_from_checkpoint:
        logger.write(f"Loading weights {args.ckpt_path}")
        ckpt_dict = torch.load(args.ckpt_path)
        try:
            model.load_state_dict(ckpt_dict)
        except:
            logger.write("Loading one-output Checkpoint")
            w = ckpt_dict["fc.weight"]
            w_ = torch.zeros((2, w.shape[1]))
            w_[1, :] = w
            b = ckpt_dict["fc.bias"]
            b_ = torch.zeros((2,))
            b_[1] = b
            ckpt_dict["fc.weight"] = w_
            ckpt_dict["fc.bias"] = b_
            model.load_state_dict(ckpt_dict)
    else:
        logger.write("Using initial weights")
    model.cuda()
    return model


def extract_data(model,loader,mask_fn,fields):
    prog=mmcv.ProgressBar(len(loader))
    all_data={ i:[] for i in fields}
    ce=torch.nn.CrossEntropyLoss(reduction= 'none')
    for b, data in enumerate(loader):
        x, labels_s, _, p = data
        x, labels_s = x.cuda(), labels_s.cuda()
        f_s,logits = model(x)
        loss=ce(logits,labels_s)
        all_param=dict(y=labels_s,f=f_s,logits=logits,loss=loss,p=p)
        cond=mask_fn(**all_param)

        if (cond).sum()>0:
            for i in fields:
                all_data[i]+=[fields[i](cond=cond,**all_param)]
        del f_s, x,all_param
        prog.update(1)
        gc.collect()

    all_data = {k: torch.vstack(v) for k, v in all_data.items()}
    return all_data

def _test_extracted(centers,all_test_data,logger,B=128):
    F,Y=all_test_data['f'],all_test_data['y']
    N=(len(Y)+B-1)//B
    PRED = None
    C=centers
    n_classes=len(C)
    prog=mmcv.ProgressBar(N)
    for b in range(N):
        f_s,labels_s = F[b*B:(b+1)*B],Y[b*B:(b+1)*B]
        dist = (f_s.unsqueeze(1) - C[None]).norm(dim=-1, p=2)
        subgroup_label = dist.argmin(dim=-1)
        pred = subgroup_label
        if PRED is None:
            PRED = pred.unsqueeze(dim=-1)
        else:
            PRED = torch.vstack([PRED, pred.unsqueeze(dim=-1)])
        prog.update(1)
        gc.collect()
    return Y,PRED

def print_acc_wga(Y,PRED,P,n_classes,logger):
    acc = lambda l, gg: ((Y != PRED) * (Y == l) * (P == gg)).sum() / ((Y == l) * (P == gg)).sum()
    accs = [(l, gg, 1 - acc(l, gg)) for l in range(n_classes) for gg in range(P.max()+1)]
    wga = min([i[-1] for i in accs])
    macc=(Y==PRED).float().mean()
    if logger is not None:
        logger.write(f"Worst Group Acc.: {wga}, Mean Acc.: {macc:.4f}")
        logger.write([f"Acc y={l}, group= {gg}.: {acc}" for l, gg, acc in accs])
        logger.write()
    return wga,macc

def _wrapped_model_test_extracted(model,all_test_data,logger,B=128,predict_method=None):
    F,Y,=all_test_data['f'],all_test_data['y'],
    N=(len(Y)+B-1)//B
    PRED = None
    prog=mmcv.ProgressBar(N)
    if predict_method is None:
        predict_method=lambda out:out.argmax(dim=-1).unsqueeze(dim=-1)
    for b in range(N):
        out =model.fc(F[b*B:(b+1)*B])
        subgroup_label = predict_method(out)
        pred = subgroup_label
        if PRED is None:
            PRED = pred
        else:
            PRED = torch.vstack([PRED, pred])
        prog.update(1)
        gc.collect()

    return Y,PRED

def wrapped_model_test_extracted(model,all_test_data,logger,B=128,predict_method=None):
    Y, PRED = _wrapped_model_test_extracted(model, all_test_data, logger, B,predict_method=predict_method)
    return print_acc_wga(Y,PRED,all_test_data['p'],model.n_classes,logger)


