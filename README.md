# Res2Net
本仓库为Res2Net模块的代码，以学习为主，原GitHub仓库为https://github.com/yfreedomliTHU/Res2Net
在学习过程中对代码进行了详细的注释



与原仓库不同的是，我的代码是单纯的res2模块，并没有结合其他的CNN网络。

细心的同学会发现代码没有初始化输入输出特征通道维度，没错，res2模块可以不写任何参数，除非有特殊需求

如果想看某一层网络的参数可以直接在相关位置加上打印代码，程序入口我已经写好了，直接写打印代码跑程序即可。
可以根据代码逻辑打印其他CNN网络中各层的特征维度参数，为网络调整带来了便携

### Architecture image
![image](https://github.com/ElegantAlan/Res2Net/blob/main/Architecture%20image/res2net.PNG?raw=true)

今天想把res2模型整合一下，发现我写的代码是真的烂透了哈哈哈

看着恶心的代码，最后还是把他搞定了。

以后写代码还是得高度模块化。
