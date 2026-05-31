import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from models.backbones.ResNet import *
from models.backbones.vit_pytorch import *
from models.backbones.vmamba import *
from models.backbones.vit_pytorch import *
from einops import rearrange


class ANmodel(torch.nn.Module):
    def __init__(self):
        super(ANmodel, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(5, 64, 3, stride=1, padding=1),  #
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 5, 3, stride=1, padding=2),  # b, 8, 3, 3
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded


def weights_init_kaiming(m):

    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block_1 = []
        # add_block_1 += [nn.Linear(input_dim, num_bottleneck)]
        add_block_1 += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block_1 += [nn.LeakyReLU(0.5)]
            #add_block_1 += [nn.SELU()]
        if dropout:
            add_block_1 += [nn.Dropout(p=0.3)]
        add_block_1 = nn.Sequential(*add_block_1)
        add_block_1.apply(weights_init_kaiming)

        # classifier_1 = []
        classifier_1 = nn.Linear(num_bottleneck, class_num)
        # classifier_1 = nn.Sequential(*classifier_1)
        classifier_1.apply(weights_init_classifier)

        self.add_block_1 = add_block_1
        self.classifier_1 = classifier_1

    def forward(self, x):
        x = self.add_block_1(x)
        x = self.classifier_1(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_init = x
        # bs x hw x c
        bs, hw, c = x.size()

        # 设置 h 和 w，使得 h * w = hw
        hh, ww = 12, 6  # 这里选择 h=8, w=9

        # spatial restore
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=ww)

        # 分割通道
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)

        # flatten
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=ww)

        # 线性层
        x = self.linear1(x)

        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        # 重新排列 x_1
        x_1 = rearrange(x_1, 'b (h w) c -> b c h w', h=hh, w=ww)
        x_1 = self.dwconv(x_1)

        # 再次展平
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=hh, w=ww)

        # 应用门控机制
        x = x_1 * x_2

        # 通过第二层线性层
        x = self.linear2(x)

        return x + x_init
    

  

class EvReId(nn.Module):
    def __init__(self, class_num=22, num_channel=None, AE_block=None):
        super().__init__()
        num_classes = class_num
        self.num_features =768
        backbone = vmamba_small_s2l15(
            num_classes=num_classes,
            imgsize=[384, 192]
            # imgsize=[256, 128]
        )
        model_path = '/data/dwj/model_path/vssm_small_0229_ckpt_epoch_222.pth'
        backbone.load_pretrained(model_path)
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.model = backbone
        self.head = nn.Linear(self.num_features, num_classes) 
        self.head2 = nn.Linear(self.num_features, num_classes) 


    def forward(self, x,datasets=None):

        """Auto Encoder"""
        voxel_reconst = None
        cash_x, cash_y = self.model(x,datasets) 
        category = self.head(cash_x)  # 线性分类
        category_y = self.head2(cash_y)  # 线性分类
        return category, category_y, cash_x, cash_x, voxel_reconst

