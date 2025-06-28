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


e1_net_seq1.pth
* trained by train_e1_seq1.py over half time interval
* curriculum training beta=0.0+0.005
* total variation regulate: 1e-2*tv_loss
* training log:
[load model from: output/v0/p_net.pth
best epoch:  49579 , min loss: 0.003088535275310278 , train time: 1963.380201101303
normalize:  0.017046787
save epoch: 0 ,loss: tensor(444.9001) ,ic: tensor(3.4936) ,res: tensor(438.5504) ,tv: tensor(285.6054) ,beta: 0.0
... RAR IC, add:  0.0 9.268856048583984
... RAR Res, add:  0.09673146158456802 117.01768493652344
save epoch: 1 ,loss: tensor(390.7776) ,ic: tensor(3.9149) ,res: tensor(383.7466) ,tv: tensor(311.6145) ,beta: 0.005
save epoch: 2 ,loss: tensor(353.3418) ,ic: tensor(4.1848) ,res: tensor(346.0798) ,tv: tensor(307.7165) ,beta: 0.01
save epoch: 5 ,loss: tensor(328.1396) ,ic: tensor(3.4788) ,res: tensor(321.8810) ,tv: tensor(277.9800) ,beta: 0.015
save epoch: 7 ,loss: tensor(307.8616) ,ic: tensor(2.5595) ,res: tensor(302.7199) ,tv: tensor(258.2125) ,beta: 0.02
save epoch: 25 ,loss: tensor(292.4333) ,ic: tensor(1.0080) ,res: tensor(288.6408) ,tv: tensor(278.4475) ,beta: 0.024999999
save epoch: 60 ,loss: tensor(277.6772) ,ic: tensor(0.3442) ,res: tensor(274.7055) ,tv: tensor(262.7448) ,beta: 0.029999997
save epoch: 77 ,loss: tensor(263.3680) ,ic: tensor(0.3790) ,res: tensor(260.5411) ,tv: tensor(244.7894) ,beta: 0.034999996
save epoch: 86 ,loss: tensor(249.5689) ,ic: tensor(0.5492) ,res: tensor(246.7371) ,tv: tensor(228.2668) ,beta: 0.039999995
save epoch: 93 ,loss: tensor(235.9253) ,ic: tensor(0.7245) ,res: tensor(232.9060) ,tv: tensor(229.4887) ,beta: 0.044999994
save epoch: 99 ,loss: tensor(223.7529) ,ic: tensor(0.7725) ,res: tensor(220.6938) ,tv: tensor(228.6619) ,beta: 0.049999993
... RAR IC, add:  0.0 3.3645131587982178
... RAR Res, add:  0.008352004922926426 98.57425689697266
save epoch: 114 ,loss: tensor(212.3073) ,ic: tensor(0.8441) ,res: tensor(209.3985) ,tv: tensor(206.4671) ,beta: 0.054999992
[load model from: output/v0/e1_net_seq1.pth
best epoch:  49573 , min loss: 0.007098150439560413 , train time: 5581.719516038895
a1 (t= 0.0 ):  0.917
0.02198647 0.01731872
a1 (t= 0.04 ):  0.888
0.027307644 0.021398727
a1 (t= 0.08 ):  0.963
0.024677001 0.026563909
a1 (t= 0.12 ):  1.136
0.03231857 0.03404656
a1 (t= 0.16 ):  1.509
0.046574354 0.038273696
a1 (t= 0.2 ):  1.673
0.066505484 0.04707188


e1_net_seq2.pth
* trained by train_e1_seq2.py over the next half time interval
* curriculum training beta=0.0+0.005
* total variation regulate: 1e-2*tv_loss
* training log:
[load model from: output/v0/p_net.pth
best epoch:  49579 , min loss: 0.003088535275310278 , train time: 1963.380201101303
[load model from: output/v0/e1_net_seq1.pth
best epoch:  49573 , min loss: 0.007098150439560413 , train time: 5581.719516038895
normalize:  0.017046787
save epoch: 0 ,loss: tensor(159.3952) ,ic: tensor(1.9871) ,res: tensor(156.5899) ,tv: tensor(81.8216) ,beta: 0.0
... RAR IC, add:  0.10000000149011612 5.80770206451416
... RAR Res, add:  0.10649897158145905 108.8899917602539
save epoch: 2 ,loss: tensor(151.2368) ,ic: tensor(2.8943) ,res: tensor(147.2615) ,tv: tensor(108.0957) ,beta: 0.005
save epoch: 3 ,loss: tensor(141.6352) ,ic: tensor(2.3467) ,res: tensor(138.2586) ,tv: tensor(102.9943) ,beta: 0.01
save epoch: 7 ,loss: tensor(133.0928) ,ic: tensor(1.2766) ,res: tensor(130.8456) ,tv: tensor(97.0699) ,beta: 0.015
save epoch: 12 ,loss: tensor(125.6054) ,ic: tensor(0.9809) ,res: tensor(123.5076) ,tv: tensor(111.6932) ,beta: 0.02
save epoch: 21 ,loss: tensor(119.2180) ,ic: tensor(0.5542) ,res: tensor(117.6094) ,tv: tensor(105.4422) ,beta: 0.024999999
save epoch: 31 ,loss: tensor(112.5737) ,ic: tensor(0.3921) ,res: tensor(111.1908) ,tv: tensor(99.0805) ,beta: 0.029999997
save epoch: 39 ,loss: tensor(106.3714) ,ic: tensor(0.4091) ,res: tensor(105.1208) ,tv: tensor(84.1484) ,beta: 0.034999996
save epoch: 46 ,loss: tensor(100.5697) ,ic: tensor(0.4400) ,res: tensor(99.2102) ,tv: tensor(91.9560) ,beta: 0.039999995
save epoch: 52 ,loss: tensor(95.4629) ,ic: tensor(0.5016) ,res: tensor(94.1358) ,tv: tensor(82.5532) ,beta: 0.044999994
save epoch: 58 ,loss: tensor(90.6316) ,ic: tensor(0.5402) ,res: tensor(89.1332) ,tv: tensor(95.8201) ,beta: 0.049999993
save epoch: 64 ,loss: tensor(85.4218) ,ic: tensor(0.5184) ,res: tensor(83.9475) ,tv: tensor(95.5933) ,beta: 0.054999992
save epoch: 69 ,loss: tensor(80.4468) ,ic: tensor(0.5386) ,res: tensor(79.1361) ,tv: tensor(77.2112) ,beta: 0.05999999
save epoch: 74 ,loss: tensor(75.7729) ,ic: tensor(0.5850) ,res: tensor(74.3013) ,tv: tensor(88.6642) ,beta: 0.06499999
save epoch: 78 ,loss: tensor(71.8400) ,ic: tensor(0.6210) ,res: tensor(70.5022) ,tv: tensor(71.6882) ,beta: 0.06999999
save epoch: 83 ,loss: tensor(67.7152) ,ic: tensor(0.6462) ,res: tensor(66.3678) ,tv: tensor(70.1278) ,beta: 0.074999996
save epoch: 88 ,loss: tensor(64.1772) ,ic: tensor(0.6566) ,res: tensor(62.8232) ,tv: tensor(69.7387) ,beta: 0.08
save epoch: 94 ,loss: tensor(60.8211) ,ic: tensor(0.6470) ,res: tensor(59.5022) ,tv: tensor(67.1893) ,beta: 0.085
... RAR IC, add:  0.10000000149011612 3.1600234508514404
... RAR Res, add:  0.11155211180448532 103.38423156738281
save epoch: 147 ,loss: tensor(57.5196) ,ic: tensor(0.5675) ,res: tensor(56.3681) ,tv: tensor(58.4041) ,beta: 0.09
save epoch: 156 ,loss: tensor(54.2738) ,ic: tensor(0.5411) ,res: tensor(53.1754) ,tv: tensor(55.7280) ,beta: 0.095000006
save epoch: 44405 ,loss: tensor(0.0106) ,ic: tensor(0.0028) ,res: tensor(0.0058) ,tv: tensor(0.1950) ,beta: 0.76999944
... RAR IC, add:  0.10000000149011612 0.2876269519329071
... RAR Res, add:  0.20000000298023224 2.5720152854919434
save epoch: 48035 ,loss: tensor(0.0099) ,ic: tensor(0.0026) ,res: tensor(0.0053) ,tv: tensor(0.1953) ,beta: 0.77499944
... RAR IC, add:  0.10000000149011612 0.22447659075260162
... RAR Res, add:  0.1529296338558197 2.9429094791412354
e1_net train complete
[load model from: output/v0/e1_net_seq2.pth
best epoch:  48035 , min loss: 0.009888878092169762 , train time: 5194.209573030472
a1 (t= 0.12 ):  0.97
0.03231857 0.040338878
a1 (t= 0.16 ):  1.147
0.046574354 0.050774705
a1 (t= 0.2 ):  1.327
0.066505484 0.060518205