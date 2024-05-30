from torch import nn
import torch.nn.functional as F
from resnet import resnet50 as resnet
import tokenizers
from transformers import BertModel, BertConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from ipdb import set_trace
from transformers import BertTokenizer
import math
from typing import Tuple, Union
from collections import OrderedDict
import numpy as np
class model_retrieval(nn.Module):
    def __init__(self,arg):
        super(model_retrieval, self).__init__()
        self.feature_extractor = resnet(pretrained=True,arg=arg)

        self.norm_layer = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.image_channel_reduce = nn.Conv2d(
          960, #3
          1,#768
          kernel_size=(1,1),
          stride=1,
          bias=False)


        self.image_patch_proj = nn.Conv2d(
          1, #3
          768,#768
          kernel_size=(7,7),
          stride=7,
          bias=False)
        self.fc1 =  nn.Linear(256, 768)
        self.fc2 =  nn.Linear(512, 768)
        self.fc3 =  nn.Linear(1024, 768)
        self.fc4 =  nn.Linear(2048, 768)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_7 = nn.MaxPool2d(kernel_size=7)
        self.maxpool_14 = nn.MaxPool2d(kernel_size=14)
        self.maxpool_28 = nn.MaxPool2d(kernel_size=28)
        self.maxpool_56 = nn.MaxPool2d(kernel_size=56)

    def forward(self, x):
    

        feature_x56, feature_x28_56, feature_x14_56, feature_x7_56= self.feature_extractor(x)
        feature_x56 = self.maxpool_56(feature_x56)# 32 64 1 1
        feature_x28_56 = self.maxpool_28(feature_x28_56)# 32 128 1 1
        feature_x14_56 = self.maxpool_14(feature_x14_56)# 32 256 1 1
        feature_x7_56 = self.avgpool(feature_x7_56)# 32 512 1 1



        # set_trace()
        f1 = self.fc1(feature_x56.view(feature_x56.shape[0],-1))
        f2 = self.fc2(feature_x28_56.view(feature_x28_56.shape[0],-1))
        f3 = self.fc3(feature_x14_56.view(feature_x14_56.shape[0],-1))
        f4 = self.fc4(feature_x7_56.view(feature_x7_56.shape[0],-1))

        features = f1 + f2 + f3 + f4


    
    ###################         


        return features,feature_x56.view(feature_x56.shape[0],-1),feature_x28_56.view(feature_x28_56.shape[0],-1),feature_x14_56.view(feature_x14_56.shape[0],-1),feature_x7_56.view(feature_x7_56.shape[0],-1)
            




class feature_map(nn.Module):
    def __init__(self,arg):
        super(feature_map, self).__init__()
    
        self.fc = nn.Linear(768, 768)
   
        self.att = nn.Linear(768, 768)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(768, 768)
        

    def forward(self,input):
        out_easy = self.fc(input)

        out_easy_att = self.sig(self.att(out_easy))
        out_easy_ = out_easy*out_easy_att
        out = self.fc2(out_easy_)
        return out_easy,out
     
#
class bertmodel(BertModel):
    def __init__(self,config,add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)

    def forward(
            self,
            input_ids=None,
            image_feature=None,
            attention_mask=None,
            key=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:  # 64*40
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if key == False:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

      
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]



            # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            
          
        return sequence_output,embedding_output
        # return 0,embedding_output
        
    




class Bert(nn.Module):
    def __init__(self, bert_name):
        super().__init__()
        # self.only_textembeddinmg = arg.only_textembeddinmg
        bert_config = BertConfig.from_pretrained(bert_name)
        # set_trace()
        self.bert_model = bertmodel.from_pretrained(bert_name,config=bert_config)
        bert_vocab = bert_name.join('/vocab.txt')
        self.tokenizers = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
        # self.fc = nn.Linear(768, 8)
        # self.fc1 = nn.Linear(768, 512)
        # self.fc2 = nn.Linear(768, 512)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,text, attention_mask):
       
            
        sequence_outputs,sequence_outputs_all = self.bert_model(input_ids=text, attention_mask=attention_mask, key=False)
        # set_trace()
        sequence_output = sequence_outputs[:, 0, :]
        # sequence_output = self.fc1(sequence_output)

        # sequence_output_embeddings = self.fc2(torch.sum(sequence_outputs_all,dim=1))
        sequence_output_embeddings = torch.sum(sequence_outputs_all,dim=1)
        
        return sequence_output,sequence_output_embeddings,sequence_outputs[:, 1:, :]
        # return 0,sequence_output_embeddings

        


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class Multi_img_crossatt(nn.Module):
    def __init__(self):
        super(Multi_img_crossatt, self).__init__()
    


        self.fc1 = nn.Linear(256, 768)
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(1024, 768)
        self.fc4 = nn.Linear(2048, 768)


 
        self.cross_attn1 = nn.MultiheadAttention(768,12)
        self.cross_attn2 = nn.MultiheadAttention(768,12)
        self.cross_attn3 = nn.MultiheadAttention(768,12)
        self.cross_attn4 = nn.MultiheadAttention(768,12)
        self.ln_pre_t = LayerNorm(768)
        self.ln_pre_i = LayerNorm(768)
        self.ln_post = LayerNorm(768)
        # init cross attn
        scale = 768**-0.5
        proj_std = scale * ((2 * 1)**-0.5)
        attn_std = scale
        nn.init.normal_(self.cross_attn1.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn1.out_proj.weight, std=proj_std)
        nn.init.normal_(self.cross_attn2.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn2.out_proj.weight, std=proj_std)
        nn.init.normal_(self.cross_attn3.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn3.out_proj.weight, std=proj_std)
        nn.init.normal_(self.cross_attn4.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn4.out_proj.weight, std=proj_std)

        self.match_fc1 = nn.Linear(768, 1)
        self.match_fc2 = nn.Linear(768, 1)
        self.match_fc3 = nn.Linear(768, 1)
        self.match_fc4 = nn.Linear(768, 1)
        # self.cls_map = nn.Linear(512, 768)
        self.cls_activation =nn.Sigmoid()

        
    # def forward(self, multi_img,text_embedding,text_cls,mask):
    def forward(self, multi_img,text_embedding,text_cls):
        # text_cls = self.cls_map(text_cls)
        multi1 = torch.unsqueeze(self.fc1(multi_img[0]),1).permute(1, 0, 2)
        multi2 = torch.unsqueeze(self.fc2(multi_img[1]),1).permute(1, 0, 2)
        multi3 = torch.unsqueeze(self.fc3(multi_img[2]),1).permute(1, 0, 2)
        multi4 = torch.unsqueeze(self.fc4(multi_img[3]),1).permute(1, 0, 2)
        text_embedding = text_embedding.permute(1, 0, 2)
     
        img_cross1 = self.cross_attn1(self.ln_pre_t(multi1),self.ln_pre_i(text_embedding),self.ln_pre_i(text_embedding),need_weights=False)[0]
        img_cross2 = self.cross_attn2(self.ln_pre_t(multi2),self.ln_pre_i(text_embedding),self.ln_pre_i(text_embedding),need_weights=False)[0]
        img_cross3 = self.cross_attn3(self.ln_pre_t(multi3),self.ln_pre_i(text_embedding),self.ln_pre_i(text_embedding),need_weights=False)[0]
        img_cross4 = self.cross_attn4(self.ln_pre_t(multi4),self.ln_pre_i(text_embedding),self.ln_pre_i(text_embedding),need_weights=False)[0]
        # set_trace()
        img_cross1 = torch.squeeze(img_cross1.permute(1, 0, 2),1)
        img_cross2 = torch.squeeze(img_cross2.permute(1, 0, 2),1)
        img_cross3 = torch.squeeze(img_cross3.permute(1, 0, 2),1)
        img_cross4 = torch.squeeze(img_cross4.permute(1, 0, 2),1)

        # set_trace()
        match1 = self.cls_activation(self.match_fc1((img_cross1+text_cls)*torch.squeeze(multi1.permute(1, 0, 2),1)))
        match2 = self.cls_activation(self.match_fc2((img_cross2+text_cls)*torch.squeeze(multi2.permute(1, 0, 2),1)))
        match3 = self.cls_activation(self.match_fc3((img_cross3+text_cls)*torch.squeeze(multi3.permute(1, 0, 2),1)))
        match4 = self.cls_activation(self.match_fc4((img_cross4+text_cls)*torch.squeeze(multi4.permute(1, 0, 2),1)))

        match = (match1,match2,match3,match4)

        
        return match












class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)














class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.b1 = ResidualAttentionBlock(width, heads, attn_mask)
        self.b2 = ResidualAttentionBlock(width, heads, attn_mask)
        self.b3 = ResidualAttentionBlock(width, heads, attn_mask)
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        # return self.resblocks(x)
        return x1,x2,x3


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # set_trace()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = x.permute(1, 0, 2)  # NLD -> LND
        x1,x2,x3 = self.transformer(x)
        x1 = x1.permute(1, 0, 2)  # LND -> NLD  
        x2 = x2.permute(1, 0, 2)
        x3 = x3.permute(1, 0, 2)

        # x = self.ln_post(x)
        x1 = self.ln_post(x1)
        x2 = self.ln_post(x2)
        x3 = self.ln_post(x3)

        # if self.proj is not None:
        #     x = x @ self.proj
    
        return x1,x2,x3





