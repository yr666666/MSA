from torch.utils.data import Dataset, DataLoader
from model import model_retrieval, feature_map
from torch.autograd import Variable
from tqdm import tqdm,trange
import random
import torch
import torch.nn as nn
import numpy as np
import numpy
import yaml
import data
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt
from ipdb import set_trace
import copy

from torch.utils.data.sampler import SubsetRandomSampler
# from tda import set_requires_grad
from model import model_retrieval, Bert,Multi_img_crossatt
from transformers import BertTokenizer
import torch.optim as optim
from transformers import BertModel, BertConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial
from eval import ratk,i2t5,t2i5
import os
import csv
import json
import time 
import matplotlib.pyplot as plt

def plot_and_save_metrics(data, title="Training Progress", ylabel="Value", filename="training_plot.png"):
    epochs = range(1, len(data) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, data, label=ylabel, color='blue', marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()  #

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_AMFMN.yaml', type=str,
                        help='path to a yaml options file') 
    # parser.add_argument('--path_opt', default='option/UCM_AMFMN.yaml', type=str,
                    # help='path to a yaml options file')
    # parser.add_argument('--path_opt', default='option/RSICD_AMFMN.yaml', type=str,
                        # help='path to a yaml options file')
    # parser.add_argument('--path_opt', default='option/SYDNEY_AMFMN.yaml', type=str,
                    # help='path to a yaml options file')

    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt) as f:
        options = yaml.safe_load(f)


    return options


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
####
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# setup_seed(1)

def element_neig_rank_loss(sim1, sim2, tem):

    
    W_h0 = torch.softmax(sim2 / tem, dim=0).t()
    W_l0 = torch.softmax(sim1.detach() / tem, dim=0).t()
  
    cross_loss0 = F.kl_div(W_l0.log(), W_h0, reduction='mean')

    knowledge_loss = cross_loss0.mean() 
    return knowledge_loss

def triplet_loss(emb_v, 
               emb_text_pos, 
               emb_text_neg, 
               emb_text, 
               emb_v_pos, 
               emb_v_neg,
               device,
              ):
    
    margin = 0.5
    # margin = 1
    alpha = 1

    v_loss_pos = 2-torch.cosine_similarity(emb_v, emb_text_pos,dim=1)
    v_loss_neg = 2-torch.cosine_similarity(emb_v, emb_text_neg,dim=1)

    t_loss_pos = 2-torch.cosine_similarity(emb_text, emb_v_pos,dim=1)
    t_loss_neg = 2-torch.cosine_similarity(emb_text, emb_v_neg,dim=1)

    triplet_loss = torch.sum(torch.max(torch.zeros(1).to(device), margin + alpha * v_loss_pos - v_loss_neg)) + torch.sum(torch.max(torch.zeros(1).to(device),margin+alpha*t_loss_pos-t_loss_neg))

    return  triplet_loss 


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_similarity_matrix_optimized(vector):
    arr = np.array(vector)
    similarity_matrix = (arr[:, np.newaxis] == arr).astype(int)
    
    return similarity_matrix



def soften(array):
    return nn.Softmax()(array/2.0)
    
def loss_func(outputs,soft_labels):#mean_square_error
    outputs=nn.Softmax()(outputs)
    soft_labels=soften(soft_labels)

    loss=-(soft_labels * torch.log(outputs)).sum()
    #print(loss)
    return loss


def main(args_re):
    torch.set_num_threads(1)
    options = parser_options()
    train_dataloader, _, test_dataloader = data.get_loaders(options["dataset"]["batch_size"], options)
    
    CrossEntropyLoss = nn.CrossEntropyLoss()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    setup_seed(1)
   
    img_model = model_retrieval(args_re).to(device)
    img_map = feature_map(args_re).to(device)
    text_model = Bert(args_re.bert_name).to(device)
    
    crossatt = Multi_img_crossatt().to(device)
 

    optimizer = optim.AdamW([{'params':text_model.parameters(),'lr':args_re.lr},
                                {'params':img_map.parameters(),'lr':args_re.lr},
                                {'params':img_model.parameters(),'lr':args_re.lr},
                                {'params':crossatt.parameters(),'lr':args_re.lr}
                               ], weight_decay=0.1,betas=(0.9, 0.999),eps=1.0e-8)

    mr = 0
    ep = 1
    tr1 = 0
    tr5 = 0
    tr10 = 0
    vr1 = 0
    vr5 = 0
    vr10 = 0
    loss1 = []
    loss2 = []
    loss3 = []
    metric = []
    for epoch in range(1, args_re.epochs + 1):

        total_loss = 0
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        total_loss_cross = 0
        nums = 0
        img_model.train()
        img_map.train()
        text_model.train()
        for step, (rs_img, text,iids) in tqdm(enumerate(train_dataloader), leave=False):

###########################################################


###############################################

            rs_img = rs_img.to(torch.float32).to(device)
            token_input_ids = []
            token_attentions = []
            cap_len = []
            for i in range(len(text)):
                # set_trace()
                token_ids = text_model.tokenizers.encode_plus(text[i],
                                                padding="max_length",
                                                max_length=31,
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                return_attention_mask=True,
                                                truncation=True
                                                )
                token_input_ids.append(token_ids['input_ids'][0])
                token_attentions.append(token_ids['attention_mask'][0])
                cap_len.append(int(token_ids['attention_mask'][0].sum().cpu().numpy()))
          
            cap_len = np.array(cap_len)
            token_ids = torch.stack(token_input_ids).to(device)
            token_attentions = torch.stack(token_attentions).to(device)
            # set_trace()
            optimizer.zero_grad()
            set_requires_grad(img_model)   



            rs_image_feature,s_1,s_2,s_3,s_4  = img_model(rs_img)
     
            _,rs_image_feature2 = img_map(rs_image_feature)

#########################################################
            # set_trace()
            text_feature,text_tokens_embeddings,sequence_outputs_all = text_model(token_ids, token_attentions)
            # set_trace()

            text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
            rs_image_feature2 = rs_image_feature2 / rs_image_feature2.norm(dim=1, keepdim=True)

 ########################################################
       # MSCMA LOSS

            out_sim = interact_cross((s_1,s_2,s_3,s_4),sequence_outputs_all,text_feature,crossatt,device)
            label_interact = torch.LongTensor(list(range(rs_img.shape[0]))).to(device)
            loss_cross1 = CrossEntropyLoss(out_sim[0], label_interact)
            loss_cross2 = CrossEntropyLoss(out_sim[1], label_interact)
            loss_cross3 = CrossEntropyLoss(out_sim[2], label_interact)
            loss_cross4 = CrossEntropyLoss(out_sim[3], label_interact)
            loss_cross = loss_cross1 + loss_cross2 + loss_cross3 + loss_cross4
  ########################################################
        # CSMMC LOSS
            ms_consis_loss = element_neig_rank_loss(out_sim[3],out_sim[2],2)\
            + element_neig_rank_loss(out_sim[3],out_sim[1],2)\
            + element_neig_rank_loss(out_sim[3],out_sim[0],2)

 ########################################################


 ########################################################  
   # triplet loss     
            adj_mat = np.eye(rs_img.shape[0])
            mask_mat_ = np.ones_like(adj_mat) - adj_mat            
            


            mask_mat = 1000000*adj_mat+mask_mat_  
            

      

            sim_it=scipy.spatial.distance.cdist(rs_image_feature2.detach().cpu().numpy(), text_feature.detach().cpu().numpy(), 'cosine')
            img_sim_mat = mask_mat*sim_it
            img_neg_text_idx = np.argmin(img_sim_mat, axis=1).astype(int)  #img_sim_mat正样本的位置都是一个很大的数
            img_neg_text = text_feature[img_neg_text_idx, :]   #将困难的文本负样本特征挑选出来
            emb_t_neg = img_neg_text
            
            sim_ti = scipy.spatial.distance.cdist(text_feature.detach().cpu().numpy(), rs_image_feature2.detach().cpu().numpy(), 'cosine')
            text_sim_mat = mask_mat*sim_ti
            text_neg_img_idx = np.argmin(text_sim_mat, axis=1).astype(int)
            text_neg_img = rs_image_feature2[text_neg_img_idx, :]   #将困难的图像负样本特征挑选出来
            emb_v_neg = text_neg_img
    
            emb_v_pos = rs_image_feature2
            emb_t_pos = text_feature

            tripletloss= triplet_loss(rs_image_feature2, 
            emb_t_pos, 
            emb_t_neg, 
            text_feature, 
            emb_v_pos, 
            emb_v_neg,
            device
            )
            loss = tripletloss

#######################################################################
       
            loss = loss + 15*loss_cross + 10*ms_consis_loss

            total_loss += loss
            loss_1 += tripletloss
            loss_2 += loss_cross
            loss_3 += ms_consis_loss
            
            loss.backward()
            optimizer.step()
            nums += 1
            # tqdm.write(f'STEP {step:03d}: loos={loss:.4f}')
            # tqdm.write(f'STEP {step:03d}: loos1={loss1:.4f}')
            # tqdm.write(f'STEP {step:03d}: loos2={loss2:.4f}')
            # tqdm.write(f'STEP {step:03d}: loos3={loss3:.4f}')
        mean_loss = total_loss / nums
        mean_loss1 = loss_1 / nums
        mean_loss2 = loss_2 / nums
        mean_loss3 = loss_3 / nums
# 

        loss1.append(mean_loss1.cpu().detach().numpy())
        loss2.append(mean_loss2.cpu().detach().numpy())
        loss3.append(mean_loss3.cpu().detach().numpy())
     
        tqdm.write(f'EPOCH {epoch:03d}: mean_loss={mean_loss:.4f}')
     
        del rs_image_feature  #Delete variables to save  memory
        del s_1 
        del s_2
        del s_3
        del s_4
        del text_feature




        # -----------------------------------------------------------------------------------
        #test
        with torch.no_grad():
            img_model.eval()
            text_model.eval()
            img_map.eval()

            image_features = []

            text_features = []
            attention_test = []
            text_ids_test = []


            for step, (rs_img, text,_) in tqdm(enumerate(test_dataloader), leave=False):
                # set_trace()
                start_time = time.time()
                rs_img = rs_img.to(torch.float32).to(device)

                rs_image_feature,s_1,s_2,s_3,s_4  = img_model(rs_img)

                ############################################################
               
                _,rs_image_feature2 = img_map(rs_image_feature)
                
                ###############################################################
                token_ids = text_model.tokenizers.encode_plus(text[0],
                                                    padding="max_length",
                                                    max_length=31,
                                                    add_special_tokens=True,
                                                    return_tensors='pt',
                                                    return_attention_mask=True,
                                                    truncation=True
                                                    )
    
                text_feature,text_embedding_test,sequence_outputs_all = text_model(token_ids['input_ids'].to(device), token_ids['attention_mask'].to(device))

                text_ids_test.append(token_ids['input_ids'][0])
                attention_test.append(token_ids['attention_mask'][0])
   
                image_features.append(rs_image_feature2[0])
 
                text_features.append(text_feature[0])
                end_time = time.time()
                elapsed_time_ms = (end_time - start_time) * 1000

                # print(f"Forward pass time: {elapsed_time_ms:.3f} ms")
    
         
            image_features = torch.stack(image_features).to(device)

            text_features = torch.stack(text_features).to(device)
         
            text_ids_test = torch.stack(text_ids_test).to(device)
            attention_test = torch.stack(attention_test).to(device)


            text_features = text_features
            image_features = image_features

            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)


######################################
            image_features = image_features.cpu().numpy().copy()
            text_features = text_features.cpu().numpy().copy()
            image_features = np.array([image_features[i] for i in range(0, len(image_features), 5)])
            t_r1, t_r5, t_r10,_,_ = i2t5(image_features,text_features)
            v_r1, v_r5, v_r10,_,_ = t2i5(image_features,text_features)
#####################################

            mean_rat = (t_r1+t_r5+t_r10+v_r1+v_r5+v_r10)/6
            if mr <= mean_rat:
                mr = mean_rat
                ep = epoch
                tr1 = t_r1
                tr5 = t_r5
                tr10 = t_r10
                vr1 = v_r1
                vr5 = v_r5
                vr10 = v_r10

                #save weight

                # best_img_model= copy.deepcopy(img_model.state_dict())
                # best_text_model= copy.deepcopy(text_model.state_dict())
                # best_img_map= copy.deepcopy(img_map.state_dict())
                # torch.save(best_img_model,'./weight/best_img_model.pth')
                # torch.save(best_text_model,'./weight/best_text_model.pth')
                # torch.save(best_img_map,'./weight/best_img_map.pth')
            metric.append(mean_rat)
            print(mr)
            print(ep)

            tqdm.write(f't_r1={t_r1}, t_r5={t_r5}, t_r10={t_r10}'
                       f'v_r1={v_r1}, v_r5={v_r5}, v_r10={v_r10}'
                       f'mr={(t_r1+t_r5+t_r10+v_r1+v_r5+v_r10)/6} ')
            del image_features  
            del text_features

    plot_and_save_metrics(loss1, title="Triplet Loss", ylabel="Loss", filename="Triplet_loss_progress.png")
    plot_and_save_metrics(loss2, title="MSCMA Loss", ylabel="Loss", filename="MSCMA_loss_progress.png")
    plot_and_save_metrics(loss3, title="CSMMC Loss", ylabel="Loss", filename="CSMMC_loss_progress.png")
    plot_and_save_metrics(metric, title="mR", ylabel="mR", filename="mR_progress_nomsnocs.png")

     # __________________________________________________________________________  
#achieve multi-scale alignment
def interact_cross(img_embeddings,text_embeddings,text_cls,model,device):
    image_features1 = torch.ones((1,256)).to(device)
    image_features2 = torch.ones((1,512)).to(device)
    image_features3 = torch.ones((1,1024)).to(device)
    image_features4 = torch.ones((1,2048)).to(device)
    text_clss = torch.ones((1,768)).to(device)
    text_features = torch.ones((1,30,768)).to(device)

    for i in range(img_embeddings[0].shape[0]):
        repeat_emb_img1 = img_embeddings[0][i,:].expand([img_embeddings[0].shape[0],256])
        repeat_emb_img2 = img_embeddings[1][i,:].expand([img_embeddings[0].shape[0],512])
        repeat_emb_img3 = img_embeddings[2][i,:].expand([img_embeddings[0].shape[0],1024])
        repeat_emb_img4 = img_embeddings[3][i,:].expand([img_embeddings[0].shape[0],2048])
        image_features1 = torch.cat((image_features1,repeat_emb_img1),0)
        image_features2 = torch.cat((image_features2,repeat_emb_img2),0)
        image_features3 = torch.cat((image_features3,repeat_emb_img3),0)
        image_features4 = torch.cat((image_features4,repeat_emb_img4),0)
        text_features = torch.cat((text_features,text_embeddings),0)
        text_clss = torch.cat((text_clss,text_cls),0)
        

    image_features1 = image_features1[1:,:]
    image_features2 = image_features2[1:,:]
    image_features3 = image_features3[1:,:]
    image_features4 = image_features4[1:,:]
    text_clss = text_clss[1:,:]
    text_features = text_features[1:,:,:]
    
    match = model((image_features1,image_features2,image_features3,image_features4),text_features,text_clss)
    
    sim1 = torch.reshape(match[0], (img_embeddings[0].shape[0],img_embeddings[0].shape[0]))
    sim2 = torch.reshape(match[1], (img_embeddings[0].shape[0],img_embeddings[0].shape[0]))
    sim3 = torch.reshape(match[2], (img_embeddings[0].shape[0],img_embeddings[0].shape[0]))
    sim4 = torch.reshape(match[3], (img_embeddings[0].shape[0],img_embeddings[0].shape[0]))

    return (sim1,sim2,sim3,sim4)



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Train a resnet on opt_name')
    arg_parser.add_argument('--epochs', type=int, default=60)
    arg_parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
    arg_parser.add_argument('--bert_name', default='../bert-base-uncased', type=str)

    args_re = arg_parser.parse_args()


    main(args_re)
    
