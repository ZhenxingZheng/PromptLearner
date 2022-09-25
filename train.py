# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from modules.regat import build_regat
from modules.get_mask import get_mask_sequence
from utils.KLLoss import KLLoss
from utils.nce_loss import NCELoss
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *
import datetime
import pandas as pd
from coop import TextEncoder
from utils.sup_contra import SupConLoss
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:,:k].float().sum()
        store.append(corrrect_k * 100.0 / batch_size)
    return store


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args, best_prec1
    global global_step
    now = str(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/sth2_ft_tem.yaml') # hmdb51, ucf101, minikinetics400, sth2
    parser.add_argument('--log_time', default=now)
    num_text_aug = 1
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    book = open(working_dir+'/'+now+'.txt', 'w')
    shutil.copy(args.config, working_dir)
    shutil.copy('train_final.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,device='cpu',jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32
    file = pd.read_csv(config.data.label_list)
    class_names = []
    for idx in range(len(file)):
        class_names.append(file.iloc[idx, 1])

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    # model_text = TextCLIP(model)
    model_text = TextEncoder(class_names, model, num_text_aug)
    model_image = ImageCLIP(model)
    guide_model = build_regat(question_dim=512, num_hid=512, reason_step=1, graph_step=1, v_dim=512, relation_dim=512, dir_num=2,
                imp_pos_emb_dim=-1, nongt_dim=20, num_heads=16, num_ans_candidates=3129,
                fusion='butd', relation_type='implicit')


    train_data = Action_DATASETS(config.data.train_list,config.data.label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.random_shift,
                       transform=transform_train, index_bias=config.data.index_bias)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS(config.data.val_list,config.data.label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val, index_bias=config.data.index_bias)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    if config.data.test_mode_augment:
        val_data = Action_DATASETS(config.data.val_list, config.data.label_list, random_shift=config.random_shift,
                                   num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                                   transform=transform_val, test_mode=True, test_mode_augment=config.data.test_mode_augment, index_bias=config.data.index_bias)
        val_loader = DataLoader(val_data,batch_size=1,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)
        fusion_model.cuda()
        guide_model.cuda()
        model_image.cuda()
        model_text.cuda()
    
    loss_img = KLLoss()
    loss_txt = KLLoss()
    # loss_classification = nn.CrossEntropyLoss()
    loss_contrast = NCELoss(temperature=0.1)
    loss_contrast = SupConLoss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            guide_model.load_state_dict(checkpoint['guide_model_state_dict'])

            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    # classes, num_text_aug, text_dict = text_prompt(train_data)
    text_dict = model_text.tokenized_prompts
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    guide_model = torch.nn.DataParallel(guide_model).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    model_text = torch.nn.DataParallel(model_text).cuda()
    
    optimizer = _optimizer(config, model, fusion_model, guide_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.data.test_mode:
        prec1, prec5 = validate(start_epoch,val_loader, device, model_image, model_text, fusion_model, config, num_text_aug)
        print(formation)
        book.write(formation + "\n")
        book.flush()
        return

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        guide_model.train()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            b,t,c,h,w = images.size()
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[i,:] for i in list_id])

            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            image_embedding = model_image(images)
            image_embedding_sequence = image_embedding.view(b,t,-1) # batch, num_frames, feature_dim
            image_embedding = fusion_model(image_embedding_sequence) # batch, feature_dim
            text_embedding, text_embedding_sequence = model_text(text_id, list_id)
            mask, q_emb_seq = get_mask_sequence(texts, text_embedding_sequence)
            composite_feature = guide_model(image_embedding_sequence.float(), q_emb_seq.float(), text_embedding.float(), mask.float())

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            
            composite_feature = F.normalize(composite_feature, dim=1)
            text_embedding = F.normalize(text_embedding, dim=1)
            twofeature = torch.cat([composite_feature.unsqueeze(1), text_embedding.unsqueeze(1)], dim=1)
            loss_contrasts = loss_contrast(twofeature, list_id)
            total_loss = (1*loss_contrasts + 1*loss_imgs + 1*loss_texts) / 3

            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        formation = str(datetime.datetime.now())+'  epoch:'+str(epoch)+'  Top1:'+str(top1.avg)+'  Top5:'+str(top5.avg)
        print(formation)

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1, prec5 = validate(epoch,val_loader, device, model_image, model_text, fusion_model, config, num_text_aug)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1, best_prec1))
        formation = str(datetime.datetime.now())+'  epoch:'+str(epoch)+'  Top1:'+str(prec1) + '  Top5:'+str(prec5)+'  Best acc:'+str(best_prec1)
        book.write(formation + "\n")
        book.flush()
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)
        epoch_saving(epoch, model, fusion_model, guide_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, guide_model, optimizer)

if __name__ == '__main__':
    main()
