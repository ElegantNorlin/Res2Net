import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#-------------------------------------------------#
#   根据MISH激活函数的公式定义MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Res2Block(nn.Module):
    def __init__(self, features_size, stride_ = 1, scale = 4, padding_ = 1, groups_ = 1, reduction = 16):
        super(Res2Block,self).__init__()
        #erro for wrong input如果输入不正确则会报错
        # features_size = 64
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        # self.divided_features = 16
        self.divided_features = int(features_size / scale)
        self.conv1 = nn.Conv2d(features_size, features_size, kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.bn1 = nn.BatchNorm2d(features_size)
        self.bn2 = nn.BatchNorm2d(self.divided_features)
		self.relu1 = nn.ReLU(inplace=True)
		self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
        self.convs = nn.ModuleList()
        

        # scale - 2 = 2循环执行两次
        for i in range(scale - 2):

            self.convs.append(
                nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
            )


    def forward(self, x):
        # x为输入特征
        # features_in.shape = torch.Size([8, 64, 32, 32])
        features_in = x
        # 这次卷积为res2模块前的那一次卷积，可以用来调整通道数，是否需要1x1卷积层根据自己网络的情况而定
        # conv1_out.shape = torch.Size([8, 64, 32, 32])
        conv1_out = self.conv1(features_in)
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.relu1(conv1_out)
        # y1为res2模块中的第一次卷积（特征没变，所以相当于没做卷积）
        # y1.shape = torch.Size([8, 16, 32, 32])
        y1 = conv1_out[:,0:self.divided_features,:,:]
        # y2.shape = torch.Size([8, 16, 32, 32])
        y2 = conv1_out[:,self.divided_features:2*self.divided_features,:,:]
        # fea为res2模块中的第二次卷积，下面用features承接了
        fea = self.conv2(y2)
        fea = self.bn2(fea)
        fea = self.relu2(fea)
        # 第二次卷积后的特征
        # 这里之所以用features变量承接是因为方便将后三次的卷积结果与第一次的卷积结果做拼接
        # features.shape = torch.Size([8, 16, 32, 32])
        features = fea
        # self.convs中只有两层网络
        for i, conv in enumerate(self.convs):
            # 第一次循环pos = 16
            # 第二次循环pos = 32
            pos = (i + 1)*self.divided_features
            # 第一次循环divided_feature.shape = torch.Size([8, 16, 32, 32])
            # 第二次循环divided_feature.shape = torch.Size([8, 16, 32, 32])
            divided_feature = conv1_out[:,pos:pos+self.divided_features,:,:]
            # 第三次和第四次卷积就是这行代码
            # 将上一次卷积结果与本次卷积的输入拼接后作为新的输入特征
            fea = conv(fea + divided_feature)
            fea = self.bn2(fea)
            fea = self.relu2(fea)
            # 下面这行代码是在此for循环完成后将后三次卷积的结果拼接在一起
            features = torch.cat([features, fea], dim = 1)
        # 将第一次的卷积和后三次卷积的结果做拼接
        out = torch.cat([y1, features], dim = 1)
        # 对拼接后的特征做1x1卷积，调整通道数
        conv1_out1 = self.conv1(out)
        result = conv1_out1 + features_in
        # 输出特征
        return result


#---------------------------------------------------#
#   CBM卷积块 = 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#   简称CBM,就是一个卷积块
#---------------------------------------------------#
class BasicConv(nn.Module):
    # in_channels输入特征通道数
    # out_channels输出特征通道数
    # kernel_size卷积核尺寸，kernel_size//2，所以卷机后长和宽仍然不变
    # stride=1 默认值为1，当调用此类若不传此参数则默认数值为1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块和yolov3中的残差块的定义类似，都是先1x1卷积，再3x3卷积   
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            Res2Block(channels)
        )

    def forward(self, x):
        return x + self.block(x)

#--------------------------------------------------------------------#
#   CSPdarknet的结构块(大的残差块)
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
class Resblock_body(nn.Module):
    # first参数是用来判断打残差块是否循环的标志
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩,其实就是步长为2的下采样
        #----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)
        # 如果大残差块不循环则调用if first中的代码
        if first:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            # 这里的卷积降维了,调整通道数
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        # 如果大残差块要循环则会调用else中的代码
        else:
            #--------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            #--------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            #----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            #----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            # 残差结构的堆叠,也就是yolov4网络结构图中特征提取网络部分每一个大的残差块的堆叠
            '''
            *[Resblock(out_channels//2) for _ in range(num_blocks)]
            为Python中的解包，可以搜索一下Python中*的用法
            在这里是*是将列表中的数值取出来一个一个来用
            '''
            # 在blocks_conv中也同时定义了残差块循环后的那个卷积层
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            # 定义了一系列残差操作完成后的卷积层
            self.concat_conv = BasicConv(out_channels, out_channels, 1)
    # 大残差块的前向传播
    # 这里建议看一下CSPX(n)模块的网络结构图（yolov4-pytorch/yolov4网络架构图/input416x416other.png）
    def forward(self, x):
        # 下采样
        x = self.downsample_conv(x)
        # 定义大残差边，也就是图CSPX(n)中的Part1部分
        x0 = self.split_conv0(x)
        # 定义n个残差块之前的卷积，也就是图CSPX(n)中的Part2部分
        x1 = self.split_conv1(x)
        # 定义了残差块循环的部分
        x1 = self.blocks_conv(x1)

        #------------------------------------#
        #   将大残差边再堆叠回来
        #------------------------------------#
        x = torch.cat([x1, x0], dim=1)
        #------------------------------------#
        #   残差拼接操作后还有一层卷积层，调整通道数最后对通道数进行整合
        #------------------------------------#
        x = self.concat_conv(x)

        return x

#---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Module):
    # layers = [1, 2, 8, 8, 4]，列表中包含了每个大的残差块循环的次数
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        # self.inplanes = 32对应了特征提取网络第一个卷积层（CBM）的输出特征通道数
        # 同时self.inplanes = 32也对应了第一个大残差块的输入特征通道数
        self.inplanes = 32
        # 第一个CBM
        # 416,416,3 -> 416,416,32
        # self.conv1.shape = torch.Size([8, 32, 416, 416])
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        # feature_channels列表定义了每个大残差块的输出特征通道数
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 整个特征提取网络的前向传播过程
    def forward(self, x):
        # self.conv1.shape = torch.Size([8, 32, 416, 416])
        x = self.conv1(x)
        # 第一轮CSP
        '''
        CSP中的下采样层.shape = torch.Size([8, 64, 208, 208])
        残差边.shape = torch.Size([8, 64, 208, 208])
        ResunitBefore.shape = torch.Size([8, 64, 208, 208])
        Resblock.shape = torch.Size([8, 64, 208, 208])
        残差循环 + 残差后的一层卷积.shape = torch.Size([8, 64, 208, 208])
        Concat.shape = torch.Size([8, 128, 208, 208])
        ConcatAfterConv.shape = torch.Size([8, 64, 208, 208])
        '''
        x = self.stages[0](x)
        # 第二轮CSP
        '''
        CSP中的下采样层.shape = torch.Size([8, 128, 104, 104])
        残差边.shape = torch.Size([8, 64, 104, 104])
        ResunitBefore.shape = torch.Size([8, 64, 104, 104])
        Resblock.shape = torch.Size([8, 64, 104, 104])
        Resblock.shape = torch.Size([8, 64, 104, 104])
        残差后的那一层卷积没有改变通道数
        Concat.shape = torch.Size([8, 128, 104, 104])
        ConcatAfterConv.shape = torch.Size([8, 128, 104, 104])
        '''
        x = self.stages[1](x)
        # 第三轮CSP
        '''
        CSP中的下采样层.shape = torch.Size([8, 256, 52, 52])
        残差边.shape = torch.Size([8, 128, 52, 52])
        ResunitBefore.shape = torch.Size([8, 128, 52, 52])
        8次循环
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        Resblock.shape = torch.Size([8, 128, 52, 52])
        残差后的那一层卷积没有改变通道数
        Concat.shape = torch.Size([8, 256, 52, 52])
        ConcatAfterConv.shape = torch.Size([8, 256, 52, 52])
        '''
        out3 = self.stages[2](x)
        # 第四轮CSP
        '''
        CSP中的下采样层.shape = torch.Size([8, 512, 26, 26])
        残差边.shape = torch.Size([8, 256, 26, 26])
        ResunitBefore.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        Resblock.shape = torch.Size([8, 256, 26, 26])
        残差后的那一层卷积没有改变通道数
        Concat.shape = torch.Size([8, 512, 26, 26])
        ConcatAfterConv.shape = torch.Size([8, 512, 26, 26])
        '''
        out4 = self.stages[3](out3)
        # 第五轮CSP
        '''
        CSP中的下采样层.shape = torch.Size([8, 1024, 13, 13])
        残差边.shape = torch.Size([8, 512, 13, 13])
        ResunitBefore.shape = torch.Size([8, 512, 13, 13])
        Resblock.shape = torch.Size([8, 512, 13, 13])
        Resblock.shape = torch.Size([8, 512, 13, 13])
        Resblock.shape = torch.Size([8, 512, 13, 13])
        Resblock.shape = torch.Size([8, 512, 13, 13])
        残差后的那一层卷积没有改变通道数
        Concat.shape = torch.Size([8, 1024, 13, 13])
        ConcatAfterConv.shape = torch.Size([8, 1024, 13, 13])
        '''
        out5 = self.stages[4](out4)
        # 返回最后三层的特征向量，以便于后续的操作
        '''
        out3.shape = torch.Size([8, 256, 52, 52])
        out4.shape = torch.Size([8, 512, 26, 26])
        out5.shape = torch.Size([8, 1024, 13, 13])
        '''
        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    # CSPDarkNet中传入的列表是特征提取网络中每个大的残差块堆叠的次数
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
  csp = darknet53(None)
  csp.cuda()
  # bs,channels,size,size
  x = Variable(torch.rand([8, 3, 416, 416]).cuda())
  y = csp(x)

