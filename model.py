import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torch.nn.parameter import Parameter

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, norm=True, bias=False, drop_rate=0.0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.drop = nn.Dropout2d(p=drop_rate) if drop_rate != 0 else None
        self.norm = nn.InstanceNorm2d(out_planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        
        x = self.conv(x)  
        if self.drop is not None:
            x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    
class GFM(nn.Module):
    def __init__(self,in_out_ch):
        super().__init__()
        self.in_out_ch = in_out_ch
        self.modality = 4
        self.fuse_conv = BasicConv(self.in_out_ch * self.modality, self.in_out_ch, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True)

    def forward(self, x):

        batch_size,c,h,w = x.size() 
        reshape_x = x.view(batch_size, self.modality, -1, 1) 
        reshape_y = reshape_x.permute(0, 3, 2, 1)  
        matrix = torch.nn.functional.cosine_similarity(
                reshape_x, reshape_y, dim=2) - torch.eye(4,device=x.device, dtype=torch.float32)
        attention_score = torch.sum(matrix * -1, dim=-1)    
        attention_prob = torch.softmax(attention_score,dim = -1)[:, :, None, None, None]
        fusion = []
        x = x.view(batch_size, 4, -1, h, w)
        for i in range(self.modality):
            fusion.append(x[:, i] * attention_prob[:,i])
        fusion = torch.cat(fusion, 1)
        fusion = self.fuse_conv(fusion)
        return fusion

class LEM(nn.Module):
    def __init__(self,dim,  kernel_size=7):
        super().__init__()
        self.dim = dim
        self.inner_dim = self.dim//2
        self.scalor = self.inner_dim ** -0.5
        self.kernel_size = kernel_size
        self.to_qkv = nn.Conv2d(in_channels=self.dim, out_channels=3 * self.inner_dim,
                             kernel_size=1, stride=1, padding=0)
        self.dwconv =  nn.Sequential(
            nn.Conv2d(self.dim,self.dim,kernel_size=7,groups=self.dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )  
        self.proj = nn.Sequential(
                nn.Conv2d(in_channels=self.inner_dim, out_channels=self.dim,
                        kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.BatchNorm2d(self.dim)
            )

    def forward(self, x):
        '''
        x: (b c h w)
        '''
        batch_size, c, h, w = x.size()
        x = x + self.dwconv(x) 
        qkv = self.to_qkv(x).reshape(batch_size, 3, self.inner_dim , -1).transpose(0, 1).contiguous() 
        q, k, v = qkv 
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        f = torch.matmul(k, v) * self.scalor
        f = F.softmax(f, dim=-1)
        res = torch.matmul(f,q)
        res = res.permute(0, 2, 1).contiguous()
        res = res.view(batch_size, self.inner_dim, *x.size()[2:])   
        res = x + self.proj(res)
        return res

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out
    

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out


class ResNet(nn.Module):
 
    def __init__(
            self,
            block,
            layers,
            num_classes=2,
            modal_size: int = 4,
            in_out_ch: int = 256,
        ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.localatt1 = LEM(dim=64 * block.expansion,  kernel_size=7)
        self.cam1_1 = nn.Conv2d(64 * block.expansion, num_classes ,kernel_size=1,stride = 1,bias=False)

        self.localatt2 = LEM(dim=128 * block.expansion,  kernel_size=7)
        self.cam2_1 = nn.Conv2d(128 * block.expansion, num_classes ,kernel_size=1,stride = 1,bias=False)

        self.localatt3 = LEM(dim=256 * block.expansion,   kernel_size=7)
        self.cam3_1 = nn.Conv2d(256 * block.expansion, num_classes ,kernel_size=1,stride = 1,bias=False)

        self.localatt4 = LEM(dim=512 * block.expansion,  kernel_size=7)
        self.cam4_1 = nn.Conv2d(512 * block.expansion, num_classes ,kernel_size=1,stride = 1,bias=False)

        self.classifier = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.cam_loss = KDLoss(temp_factor = 3)

        self.GF_modules = nn.ModuleList()
        for _ in range(modal_size):
            self.GF_modules.append(
                GFM(in_out_ch)
            )
            in_out_ch *= 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        batch_size, modal_num, C, W, H = x.size()

        x = x.view(batch_size * modal_num, C, W, H)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        layer1_x = self.layer1(x)
        seg_feature1 = self.localatt1(layer1_x)  
        seg_feature1 = seg_feature1.view(batch_size, modal_num * seg_feature1.size(1), *seg_feature1.size()[2:])
        seg_feature1 = self.GF_modules[0](seg_feature1)
        seg_feature1 = self.cam1_1(seg_feature1)
        middle_cam1 = self.avgpool(seg_feature1)
        out1 = self.classifier(middle_cam1).view(middle_cam1.size(0), -1)


        layer2_x = self.layer2(layer1_x)
        seg_feature2 = self.localatt2(layer2_x)
        seg_feature2 = seg_feature2.view(batch_size, modal_num * seg_feature2.size(1), *seg_feature2.size()[2:])
        seg_feature2 = self.GF_modules[1](seg_feature2)
        seg_feature2 = self.cam2_1(seg_feature2)   
        middle_cam2 = self.avgpool(seg_feature2)
        out2 = self.classifier(middle_cam2).view(middle_cam2.size(0), -1)

       
        layer3_x = self.layer3(layer2_x)
        seg_feature3 = self.localatt3(layer3_x)
        seg_feature3 = seg_feature3.view(batch_size, modal_num * seg_feature3.size(1), *seg_feature3.size()[2:])
        seg_feature3 = self.GF_modules[2](seg_feature3)
        seg_feature3 = self.cam3_1(seg_feature3)
        middle_cam3 = self.avgpool(seg_feature3)
        out3 = self.classifier(middle_cam3).view(middle_cam3.size(0), -1)
  

        layer4_x = self.layer4(layer3_x)
        seg_feature4 = self.localatt4(layer4_x)
        seg_feature4 = seg_feature4.view(batch_size, modal_num * seg_feature4.size(1), *seg_feature4.size()[2:])
        seg_feature4 = self.GF_modules[3](seg_feature4)
        middle_cam4 = self.cam4_1(seg_feature4)
        out4 = self.classifier(middle_cam4).view(middle_cam4.size(0), -1)

        camloss = self.cam_loss(out1, out4)+self.cam_loss(out2, out4)+ self.cam_loss(out3, out4)

        return out4,out1,out2,out3,camloss


def resnet18(pretrained=True, **kwargs):
    """ return a ResNet 18 object
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        #pretrained_dict = torch.load('../cnn/checkpoints/resnet50_with_twoout--79--best.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
      
    return model

def resnet34(pretrained=True, **kwargs):
    """ return a ResNet 34 object
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
     
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
      
    return model

def resnet101(pretrained=True, **kwargs):
    """ return a ResNet 101 object
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
     
    return model


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    net = resnet101(pretrained= False)
    x = torch.randn(size=(1, 4,3,224, 224))
    out,a,b,c,d = net(x)
    print(out.shape)
