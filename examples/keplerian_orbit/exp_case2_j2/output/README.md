# v0

## train p_net
* training log:
p net scale:  0.18991181
save epoch: 0 ,loss: tensor(2.7721) ,ic: tensor(1.7133) ,res: tensor(1.0341) ,tv: tensor(2.4719) ,beta:  0.0
... RAR IC, add:  0.0 3.8275725841522217
... RAR Res, add:  0.020027311518788338 68.31773376464844
save epoch: 1 ,loss: tensor(1.4248) ,ic: tensor(0.9195) ,res: tensor(0.4873) ,tv: tensor(1.7980) ,beta:  0.02
save epoch: 2 ,loss: tensor(0.7899) ,ic: tensor(0.4309) ,res: tensor(0.3432) ,tv: tensor(1.5765) ,beta:  0.04
save epoch: 3 ,loss: tensor(0.4395) ,ic: tensor(0.1955) ,res: tensor(0.2300) ,tv: tensor(1.3949) ,beta:  0.06
save epoch: 4 ,loss: tensor(0.2529) ,ic: tensor(0.0981) ,res: tensor(0.1436) ,tv: tensor(1.1116) ,beta:  0.08
save epoch: 5 ,loss: tensor(0.1550) ,ic: tensor(0.0636) ,res: tensor(0.0836) ,tv: tensor(0.7838) ,beta:  0.099999994
save epoch: 6 ,loss: tensor(0.1051) ,ic: tensor(0.0539) ,res: tensor(0.0461) ,tv: tensor(0.5055) ,beta:  0.11999999
save epoch: 7 ,loss: tensor(0.0809) ,ic: tensor(0.0528) ,res: tensor(0.0250) ,tv: tensor(0.3122) ,beta:  0.13999999
save epoch: 8 ,loss: tensor(0.0698) ,ic: tensor(0.0542) ,res: tensor(0.0137) ,tv: tensor(0.1918) ,beta:  0.15999998
save epoch: 9 ,loss: tensor(0.0651) ,ic: tensor(0.0561) ,res: tensor(0.0078) ,tv: tensor(0.1200) ,beta:  0.17999998
... RAR IC, add:  0.0 1.0642411708831787
save epoch: 379 ,loss: tensor(0.0618) ,ic: tensor(0.0609) ,res: tensor(0.0007) ,tv: tensor(0.0282) ,beta:  0.19999997
... RAR IC, add:  0.0 1.0284017324447632
... RAR Res, add:  0.199101984500885 0.878056526184082
save epoch: 413 ,loss: tensor(0.0586) ,ic: tensor(0.0563) ,res: tensor(0.0016) ,tv: tensor(0.0622) ,beta:  0.21999997
save epoch: 47521 ,loss: tensor(0.0031) ,ic: tensor(0.0010) ,res: tensor(0.0006) ,tv: tensor(0.1485) ,beta:  1.0
... RAR Res, add:  0.18251702189445496 0.14605022966861725
save epoch: 49652 ,loss: tensor(0.0029) ,ic: tensor(0.0009) ,res: tensor(0.0005) ,tv: tensor(0.1441) ,beta:  1.0
... RAR Res, add:  0.15943610668182373 0.15062402188777924
p_net_reg train complete
[load model] from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936
[test] max(p_mc - p_nn) at t=0.000: 0.0176
[test] max(p_mc - p_nn) at t=0.040: 0.0217
[test] max(p_mc - p_nn) at t=0.080: 0.0257
[test] max(p_mc - p_nn) at t=0.120: 0.0350
[test] max(p_mc - p_nn) at t=0.160: 0.0522
[test] max(p_mc - p_nn) at t=0.200: 0.0665

## train e1_net_seq1.pth
* training log:
[load model] from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936
normalize:  0.017596237
save epoch: 0 ,loss: tensor(531.5310) ,ic: tensor(8.5108) ,res: tensor(520.0043) ,tv: tensor(301.5949) ,beta: 0.000
save epoch: 1 ,loss: tensor(352.9469) ,ic: tensor(7.3731) ,res: tensor(342.6255) ,tv: tensor(294.8325) ,beta: 0.005
save epoch: 2 ,loss: tensor(327.0011) ,ic: tensor(6.4037) ,res: tensor(317.9422) ,tv: tensor(265.5270) ,beta: 0.010
save epoch: 6 ,loss: tensor(293.4273) ,ic: tensor(4.2649) ,res: tensor(286.6021) ,tv: tensor(256.0332) ,beta: 0.015
save epoch: 7 ,loss: tensor(278.4329) ,ic: tensor(3.9922) ,res: tensor(272.2419) ,tv: tensor(219.8783) ,beta: 0.020
save epoch: 23 ,loss: tensor(264.2774) ,ic: tensor(2.1431) ,res: tensor(259.5493) ,tv: tensor(258.4996) ,beta: 0.025
save epoch: 51 ,loss: tensor(250.7235) ,ic: tensor(0.8552) ,res: tensor(247.7799) ,tv: tensor(208.8331) ,beta: 0.030
save epoch: 74 ,loss: tensor(237.6845) ,ic: tensor(0.6470) ,res: tensor(235.0580) ,tv: tensor(197.9496) ,beta: 0.035
save epoch: 86 ,loss: tensor(225.0248) ,ic: tensor(0.5808) ,res: tensor(222.2234) ,tv: tensor(222.0497) ,beta: 0.040
save epoch: 94 ,loss: tensor(213.1414) ,ic: tensor(0.5883) ,res: tensor(210.4201) ,tv: tensor(213.2993) ,beta: 0.045
... RAR IC, add:  0.0 3.209878444671631
... RAR Res, add:  0.0229537021368742 97.9327163696289
save epoch: 116 ,loss: tensor(201.4691) ,ic: tensor(0.7113) ,res: tensor(198.7306) ,tv: tensor(202.7111) ,beta: 0.050
save epoch: 124 ,loss: tensor(191.3023) ,ic: tensor(0.6786) ,res: tensor(188.8308) ,tv: tensor(179.2860) ,beta: 0.055
...
save epoch: 47404 ,loss: tensor(0.0056) ,ic: tensor(0.0012) ,res: tensor(0.0033) ,tv: tensor(0.1135) ,beta: 0.885
... RAR IC, add:  0.0 0.14512450993061066
... RAR Res, add:  0.09203396737575531 0.6644783020019531
[load model] from: output/v0/e1_net_seq1.pth
best epoch:  47404 , min loss: 0.005601462908089161 , train time: 19477.8390481472
[check] x1 ranges, true joint pdf shape, type:  float32 (50, 50, 50, 50) float32
a1 (t= 0.0 ):  0.137
[test] max(p_mc - p_nn) at t=0.000: 0.0176
[test] max(p_mc - p_nn), max(e1_nn) at t=0.000: 0.0176 vs 0.0173
a1 (t= 0.04 ):  0.903
[test] max(p_mc - p_nn) at t=0.040: 0.0217
[test] max(p_mc - p_nn), max(e1_nn) at t=0.040: 0.0217 vs 0.0205
a1 (t= 0.08 ):  1.034
[test] max(p_mc - p_nn) at t=0.080: 0.0257
[test] max(p_mc - p_nn), max(e1_nn) at t=0.080: 0.0257 vs 0.0259
a1 (t= 0.12 ):  1.254
[test] max(p_mc - p_nn) at t=0.120: 0.0350
[test] max(p_mc - p_nn), max(e1_nn) at t=0.120: 0.0350 vs 0.0325
a1 (t= 0.16 ):  1.509
[test] max(p_mc - p_nn) at t=0.160: 0.0522
[test] max(p_mc - p_nn), max(e1_nn) at t=0.160: 0.0522 vs 0.0389
a1 (t= 0.2 ):  1.716
[test] max(p_mc - p_nn) at t=0.200: 0.0665
[test] max(p_mc - p_nn), max(e1_nn) at t=0.200: 0.0665 vs 0.0452

## train e1_net_seq2.pth
* training log:
[load model] from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936
[load model] from: output/v0/e1_net_seq1.pth
best epoch:  47404 , min loss: 0.005601462908089161 , train time: 19477.8390481472
normalize:  0.017596237
save epoch: 0 ,loss: tensor(457.4837) ,ic: tensor(13.0377) ,res: tensor(443.6584) ,tv: tensor(78.7637) ,beta: 0.0
save epoch: 1 ,loss: tensor(179.2849) ,ic: tensor(9.0442) ,res: tensor(169.4146) ,tv: tensor(82.6179) ,beta: 0.005
save epoch: 5 ,loss: tensor(168.5805) ,ic: tensor(4.7414) ,res: tensor(162.7724) ,tv: tensor(106.6761) ,beta: 0.01
save epoch: 6 ,loss: tensor(136.6887) ,ic: tensor(4.4582) ,res: tensor(131.2943) ,tv: tensor(93.6184) ,beta: 0.015
save epoch: 7 ,loss: tensor(126.3153) ,ic: tensor(4.2810) ,res: tensor(121.0703) ,tv: tensor(96.3950) ,beta: 0.02
save epoch: 14 ,loss: tensor(116.4890) ,ic: tensor(2.5443) ,res: tensor(112.9960) ,tv: tensor(94.8728) ,beta: 0.024999999
save epoch: 23 ,loss: tensor(109.0437) ,ic: tensor(1.3067) ,res: tensor(106.7495) ,tv: tensor(98.7457) ,beta: 0.029999997
save epoch: 33 ,loss: tensor(103.4953) ,ic: tensor(0.8309) ,res: tensor(101.8085) ,tv: tensor(85.5909) ,beta: 0.034999996
save epoch: 62 ,loss: tensor(98.1856) ,ic: tensor(0.4914) ,res: tensor(96.9290) ,tv: tensor(76.5137) ,beta: 0.039999995
save epoch: 100 ,loss: tensor(93.1837) ,ic: tensor(0.3967) ,res: tensor(92.0435) ,tv: tensor(74.3452) ,beta: 0.044999994
... RAR IC, add:  0.10000000149011612 2.501511812210083
... RAR Res, add:  0.11079372465610504 95.14533233642578
save epoch: 176 ,loss: tensor(88.4313) ,ic: tensor(0.5117) ,res: tensor(87.0892) ,tv: tensor(83.0380) ,beta: 0.049999993
save epoch: 188 ,loss: tensor(83.6231) ,ic: tensor(0.5551) ,res: tensor(82.3073) ,tv: tensor(76.0724) ,beta: 0.054999992
save epoch: 199 ,loss: tensor(79.3104) ,ic: tensor(0.6311) ,res: tensor(77.8733) ,tv: tensor(80.5986) ,beta: 0.05999999
... RAR IC, add:  0.10000000149011612 3.6978697776794434
... RAR Res, add:  0.1820114552974701 93.0307388305664
save epoch: 242 ,loss: tensor(75.0282) ,ic: tensor(0.8430) ,res: tensor(73.3737) ,tv: tensor(81.1557) ,beta: 0.06499999
save epoch: 49092 ,loss: tensor(0.0101) ,ic: tensor(0.0023) ,res: tensor(0.0059) ,tv: tensor(0.1925) ,beta: 0.8049994
... RAR IC, add:  0.10000000149011612 0.2604706883430481
... RAR Res, add:  0.10176988691091537 1.2834733724594116
[load model] from: output/v0/e1_net_seq2.pth
best epoch:  49092 , min loss: 0.010128185153007507 , train time: 15809.360105276108
a1 (t= 0.12 ):  1.1
[test] max(p_mc - p_nn) at t=0.120: 0.0350
[test] max(p_mc - p_nn), max(e1_nn) at t=0.120: 0.0350 vs 0.0367
a1 (t= 0.16 ):  1.194
[test] max(p_mc - p_nn) at t=0.160: 0.0522
[test] max(p_mc - p_nn), max(e1_nn) at t=0.160: 0.0522 vs 0.0492
a1 (t= 0.2 ):  1.383
[test] max(p_mc - p_nn) at t=0.200: 0.0665
[test] max(p_mc - p_nn), max(e1_nn) at t=0.200: 0.0665 vs 0.0555


# base
## train p_net.pth
* training log:
p net scale:  0.18991181
save epoch: 0 ,loss: tensor(162.1012) ,ic: tensor(1.7133) ,res: tensor(160.3878) ,beta:  1.0
... RAR IC, add:  0.0 3.8275725841522217
... RAR Res, add:  0.020027311518788338 68.31773376464844
save epoch: 1 ,loss: tensor(97.6336) ,ic: tensor(0.9020) ,res: tensor(96.7316) ,beta:  1.0
save epoch: 2 ,loss: tensor(48.1197) ,ic: tensor(0.4176) ,res: tensor(47.7021) ,beta:  1.0
save epoch: 3 ,loss: tensor(20.3430) ,ic: tensor(0.1811) ,res: tensor(20.1619) ,beta:  1.0
save epoch: 4 ,loss: tensor(7.4848) ,ic: tensor(0.0865) ,res: tensor(7.3983) ,beta:  1.0
save epoch: 5 ,loss: tensor(2.5519) ,ic: tensor(0.0559) ,res: tensor(2.4960) ,beta:  1.0
save epoch: 6 ,loss: tensor(0.8868) ,ic: tensor(0.0488) ,res: tensor(0.8380) ,beta:  1.0
save epoch: 7 ,loss: tensor(0.3474) ,ic: tensor(0.0492) ,res: tensor(0.2982) ,beta:  1.0
save epoch: 8 ,loss: tensor(0.1681) ,ic: tensor(0.0514) ,res: tensor(0.1167) ,beta:  1.0
save epoch: 9 ,loss: tensor(0.1049) ,ic: tensor(0.0537) ,res: tensor(0.0512) ,beta:  1.0
save epoch: 10 ,loss: tensor(0.0810) ,ic: tensor(0.0557) ,res: tensor(0.0254) ,beta:  1.0
save epoch: 11 ,loss: tensor(0.0714) ,ic: tensor(0.0573) ,res: tensor(0.0141) ,beta:  1.0
save epoch: 12 ,loss: tensor(0.0673) ,ic: tensor(0.0586) ,res: tensor(0.0087) ,beta:  1.0
... RAR IC, add:  0.0 1.0632961988449097
save epoch: 48202 ,loss: tensor(0.0024) ,ic: tensor(0.0007) ,res: tensor(0.0017) ,beta:  1.0
... RAR Res, add:  0.17783498764038086 0.2499466836452484
save epoch: 49227 ,loss: tensor(0.0023) ,ic: tensor(0.0006) ,res: tensor(0.0016) ,beta:  1.0
... RAR Res, add:  0.1127374917268753 0.2305828183889389
[load model] from: output/base/p_net.pth
best epoch:  49227 , min loss: 0.0022764920722693205 , train time: 4110.011565208435
[test] pdf(NN) marginalized to rphi at t= 0.0
[test] pdf(NN) marginalized to rphi at t= 0.04
[test] pdf(NN) marginalized to rphi at t= 0.08
[test] pdf(NN) marginalized to rphi at t= 0.12
[test] pdf(NN) marginalized to rphi at t= 0.16
[test] pdf(NN) marginalized to rphi at t= 0.2
[test] pdf(nn) marginalized to xy at t'= 0.0
[check] sum p_nn (N-sphere):  0.9047998
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.90 and (x,y) 0.90
[test] pdf(nn) marginalized to xy at t'= 0.04
[check] sum p_nn (N-sphere):  0.90773237
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.91 and (x,y) 0.91
[test] pdf(nn) marginalized to xy at t'= 0.08
[check] sum p_nn (N-sphere):  0.90942454
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.91 and (x,y) 0.91
[test] pdf(nn) marginalized to xy at t'= 0.12
[check] sum p_nn (N-sphere):  0.9043019
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.90 and (x,y) 0.90
[test] pdf(nn) marginalized to xy at t'= 0.16
[check] sum p_nn (N-sphere):  0.89936453
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.90 and (x,y) 0.90
[test] pdf(nn) marginalized to xy at t'= 0.2
[check] sum p_nn (N-sphere):  0.9053106
[check] p_monte in (x,y) 1.00 & p_nn sum in (r,phi) 0.91 and (x,y) 0.91
[test] max(p_mc - p_nn) at t=0.000: 0.0155
[test] max(p_mc - p_nn) at t=0.040: 0.0197
[test] max(p_mc - p_nn) at t=0.080: 0.0277
[test] max(p_mc - p_nn) at t=0.120: 0.0372
[test] max(p_mc - p_nn) at t=0.160: 0.0581
[test] max(p_mc - p_nn) at t=0.200: 0.0687