#Res2Net
from __future__ import division
import torch
import torch.nn as nn
import os
from torch.autograd import Variable

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# res2模块的输入和输出特征维度是相等的，我已经测试过了，也可以自己测试看一下
'''
pytorch中tensor的维度分别代表什么？
注释中的参数是以下面的例子讲解的！
举个例子：torch.Size([8, 64, 32, 32])
8为batch_size
64为channels
32为宽
32为高
features_size即是输入特征尺寸也是输出特征尺寸，每个尺度的特征提取features_size都不同
着我里是因为我的代码原因，我需要设置成这样，建议大家可以动态的传入参数
怎么样动态的传入特征输入、输出维度的参数呢？
其实很简单，不需要修改代码，在调用Res2Net类的时候只传入第一个参数，也就是features_size参数
'''
class Res2Net(nn.Module):
    def __init__(self, features_size参数 = 64, stride_ = 1, scale = 4, padding_ = 1, groups_ = 1, reduction = 16):
        super(Res2Net,self).__init__()
        #erro for wrong input如果输入不正确则会报错
        # features_size = 64
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        # self.divided_features = 16
        self.divided_features = int(features_size / scale)
        self.conv1 = nn.Conv2d(features_size, features_size, kernel_size=1, stride=stride_, padding=0, groups=groups_)
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
        # y1为res2模块中的第一次卷积（特征没变，所以相当于没做卷积）
        # y1.shape = torch.Size([8, 16, 32, 32])
        y1 = conv1_out[:,0:self.divided_features,:,:]
        # y2.shape = torch.Size([8, 16, 32, 32])
        y2 = conv1_out[:,self.divided_features:2*self.divided_features,:,:]
        # fea为res2模块中的第二次卷积，下面用features承接了
        fea = self.conv2(y2)
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
            # 下面这行代码是在此for循环完成后将后三次卷积的结果拼接在一起
            features = torch.cat([features, fea], dim = 1)
        # 将第一次的卷积和后三次卷积的结果做拼接
        out = torch.cat([y1, features], dim = 1)
        # 对拼接后的特征做1x1卷积，调整通道数
        conv1_out1 = self.conv1(out)
        result = conv1_out1
        # 输出特征
        return result

if __name__ == "__main__":
    res2net = Res2Net(64,1,4,1,1,16)
    res2net.cuda()
    # bs,channels,height,width
    x = Variable(torch.rand([8, 64, 32, 32]).cuda())
    y = res2net(x)
    # x.shape = torch.Size([8, 64, 32, 32])
    print(x.shape)
    # y.shape = torch.Size([8, 64, 32, 32])
    print(y.shape)
    print(res2net)
    torch.save(res2net, 'Res2Net.pth')
