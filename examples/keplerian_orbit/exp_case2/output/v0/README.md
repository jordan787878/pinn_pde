v0/
    
p_net.pth
* trained by train_p.py
* curriculum training beta=0.0+0.02
* total variation regulate: 1e-2*tv_loss
* training log:
p net scale:  0.18991181
save epoch: 0 ,loss: tensor(2.7721) ,ic: tensor(1.7133) ,res: tensor(1.0341) ,tv: tensor(2.4719) ,beta:  0.0
... RAR IC, add:  0.0 3.8275725841522217
... RAR Res, add:  0.020027311518788338 68.32891845703125
save epoch: 1 ,loss: tensor(1.4248) ,ic: tensor(0.9195) ,res: tensor(0.4873) ,tv: tensor(1.7979) ,beta:  0.02
save epoch: 2 ,loss: tensor(0.7899) ,ic: tensor(0.4309) ,res: tensor(0.3432) ,tv: tensor(1.5765) ,beta:  0.04
save epoch: 3 ,loss: tensor(0.4395) ,ic: tensor(0.1955) ,res: tensor(0.2300) ,tv: tensor(1.3948) ,beta:  0.06
save epoch: 4 ,loss: tensor(0.2529) ,ic: tensor(0.0981) ,res: tensor(0.1436) ,tv: tensor(1.1115) ,beta:  0.08
save epoch: 5 ,loss: tensor(0.1550) ,ic: tensor(0.0636) ,res: tensor(0.0836) ,tv: tensor(0.7837) ,beta:  0.099999994
save epoch: 6 ,loss: tensor(0.1051) ,ic: tensor(0.0539) ,res: tensor(0.0461) ,tv: tensor(0.5054) ,beta:  0.11999999
save epoch: 7 ,loss: tensor(0.0809) ,ic: tensor(0.0528) ,res: tensor(0.0250) ,tv: tensor(0.3121) ,beta:  0.13999999
save epoch: 8 ,loss: tensor(0.0699) ,ic: tensor(0.0542) ,res: tensor(0.0137) ,tv: tensor(0.1918) ,beta:  0.15999998
save epoch: 9 ,loss: tensor(0.0651) ,ic: tensor(0.0561) ,res: tensor(0.0078) ,tv: tensor(0.1200) ,beta:  0.17999998
... RAR IC, add:  0.0 1.0642411708831787
save epoch: 379 ,loss: tensor(0.0618) ,ic: tensor(0.0609) ,res: tensor(0.0007) ,tv: tensor(0.0282) ,beta:  0.19999997
... RAR IC, add:  0.0 1.028399109840393
... RAR Res, add:  0.199101984500885 0.8782105445861816
save epoch: 413 ,loss: tensor(0.0586) ,ic: tensor(0.0563) ,res: tensor(0.0016) ,tv: tensor(0.0622) ,beta:  0.21999997
save epoch: 426 ,loss: tensor(0.0556) ,ic: tensor(0.0522) ,res: tensor(0.0026) ,tv: tensor(0.0780) ,beta:  0.23999996
save epoch: 446 ,loss: tensor(0.0527) ,ic: tensor(0.0480) ,res: tensor(0.0037) ,tv: tensor(0.1032) ,beta:  0.25999996
save epoch: 479 ,loss: tensor(0.0500) ,ic: tensor(0.0465) ,res: tensor(0.0030) ,tv: tensor(0.0580) ,beta:  0.27999997
... RAR IC, add:  0.0 0.9148398041725159
... RAR Res, add:  0.18391795456409454 1.006196141242981
save epoch: 590 ,loss: tensor(0.0475) ,ic: tensor(0.0456) ,res: tensor(0.0015) ,tv: tensor(0.0363) ,beta:  0.29999998
...
save epoch: 47054 ,loss: tensor(0.0033) ,ic: tensor(0.0012) ,res: tensor(0.0006) ,tv: tensor(0.1500) ,beta:  1.0
... RAR Res, add:  0.17846857011318207 0.25211480259895325
save epoch: 49579 ,loss: tensor(0.0031) ,ic: tensor(0.0011) ,res: tensor(0.0006) ,tv: tensor(0.1451) ,beta:  1.0
... RAR Res, add:  0.177302747964859 0.18425658345222473
p_net_reg train complete
[load model from: output/v0/p_net.pth
best epoch:  49579 , min loss: 0.003088535275310278 , train time: 1963.380201101303


e1_net.pth
* trained by train_e1_seq1.py over entire time interval
* curriculum training beta=0.0+0.005
* total variation regulate: 1e-2*tv_loss
* training log:
[load model from: output/v0/p_net.pth
best epoch:  49579 , min loss: 0.003088535275310278 , train time: 1963.380201101303
normalize:  0.017046787
save epoch: 0 ,loss: tensor(361.7794) ,ic: tensor(3.4958) ,res: tensor(356.4157) ,tv: tensor(186.7868) ,beta: 0.0
... RAR IC, add:  0.0 9.268856048583984
... RAR Res, add:  0.0344587080180645 112.5899887084961
save epoch: 1 ,loss: tensor(306.4502) ,ic: tensor(3.9044) ,res: tensor(300.5685) ,tv: tensor(197.7250) ,beta: 0.005
save epoch: 2 ,loss: tensor(266.0138) ,ic: tensor(4.1682) ,res: tensor(259.8362) ,tv: tensor(200.9389) ,beta: 0.01
save epoch: 6 ,loss: tensor(246.7687) ,ic: tensor(2.7378) ,res: tensor(242.1330) ,tv: tensor(189.7847) ,beta: 0.015
save epoch: 8 ,loss: tensor(228.0297) ,ic: tensor(1.8968) ,res: tensor(224.3254) ,tv: tensor(180.7564) ,beta: 0.02
save epoch: 21 ,loss: tensor(216.5858) ,ic: tensor(1.1386) ,res: tensor(213.6868) ,tv: tensor(176.0353) ,beta: 0.024999999
save epoch: 44 ,loss: tensor(205.6857) ,ic: tensor(0.5644) ,res: tensor(203.4790) ,tv: tensor(164.2189) ,beta: 0.029999997
save epoch: 79 ,loss: tensor(195.1169) ,ic: tensor(0.3335) ,res: tensor(193.2465) ,tv: tensor(153.6863) ,beta: 0.034999996
save epoch: 100 ,loss: tensor(185.1514) ,ic: tensor(0.3897) ,res: tensor(183.2712) ,tv: tensor(149.0499) ,beta: 0.039999995
... RAR IC, add:  0.0 2.213921546936035
... RAR Res, add:  0.053611721843481064 112.02901458740234
save epoch: 132 ,loss: tensor(175.2901) ,ic: tensor(0.6316) ,res: tensor(172.8119) ,tv: tensor(184.6672) ,beta: 0.044999994
save epoch: 45016 ,loss: tensor(0.0145) ,ic: tensor(0.0025) ,res: tensor(0.0087) ,tv: tensor(0.3269) ,beta: 0.8049994
... RAR IC, add:  0.0 0.21896842122077942
... RAR Res, add:  0.20000000298023224 2.618964672088623
save epoch: 47666 ,loss: tensor(0.0137) ,ic: tensor(0.0023) ,res: tensor(0.0084) ,tv: tensor(0.2999) ,beta: 0.8099994
... RAR IC, add:  0.0 0.19820375740528107
... RAR Res, add:  0.19384297728538513 1.5692147016525269
e1_net train complete
[load model from: output/v0/e1_net_seq1.pth
best epoch:  47666 , min loss: 0.013721784576773643 , train time: 4692.972725152969
a1 (t= 0.0 ):  0.92
0.02198647 0.016432593
a1 (t= 0.04 ):  0.771
0.027307644 0.024692224
a1 (t= 0.08 ):  0.781
0.024677001 0.03545536
a1 (t= 0.12 ):  0.972
0.03231857 0.046268117
a1 (t= 0.16 ):  1.044
0.046574354 0.05578202
a1 (t= 0.2 ):  1.244
0.066505484 0.06494533