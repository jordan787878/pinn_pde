# v0

# train p_net
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
[load model from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936

# train e1_net_seq1.pth
# training log:
[load model from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936
normalize:  0.017596237
save epoch: 0 ,loss: tensor(428.0074) ,ic: tensor(3.4771) ,res: tensor(421.7364) ,tv: tensor(279.3858) ,beta: 0.0
... RAR IC, add:  0.0 9.268856048583984
... RAR Res, add:  0.09673146158456802 114.33500671386719
save epoch: 1 ,loss: tensor(373.5867) ,ic: tensor(3.8843) ,res: tensor(366.6864) ,tv: tensor(301.6060) ,beta: 0.005
save epoch: 2 ,loss: tensor(337.0647) ,ic: tensor(4.1373) ,res: tensor(329.9049) ,tv: tensor(302.2522) ,beta: 0.01
save epoch: 5 ,loss: tensor(311.8399) ,ic: tensor(3.4169) ,res: tensor(305.6985) ,tv: tensor(272.4572) ,beta: 0.015
save epoch: 7 ,loss: tensor(292.2715) ,ic: tensor(2.5137) ,res: tensor(287.2087) ,tv: tensor(254.9028) ,beta: 0.02
save epoch: 25 ,loss: tensor(277.5602) ,ic: tensor(0.9736) ,res: tensor(273.9180) ,tv: tensor(266.8602) ,beta: 0.024999999
save epoch: 60 ,loss: tensor(263.5288) ,ic: tensor(0.3282) ,res: tensor(260.6467) ,tv: tensor(255.3908) ,beta: 0.029999997
save epoch: 77 ,loss: tensor(250.3523) ,ic: tensor(0.3734) ,res: tensor(247.5921) ,tv: tensor(238.6749) ,beta: 0.034999996
save epoch: 87 ,loss: tensor(236.8586) ,ic: tensor(0.5751) ,res: tensor(233.9960) ,tv: tensor(228.7522) ,beta: 0.039999995
save epoch: 94 ,loss: tensor(224.4548) ,ic: tensor(0.7454) ,res: tensor(221.5108) ,tv: tensor(219.8641) ,beta: 0.044999994
save epoch: 100 ,loss: tensor(213.1911) ,ic: tensor(0.7801) ,res: tensor(210.2498) ,tv: tensor(216.1263) ,beta: 0.049999993
... RAR IC, add:  0.0 3.328810453414917
... RAR Res, add:  0.008352004922926426 95.1323471069336
save epoch: 115 ,loss: tensor(202.2834) ,ic: tensor(0.8247) ,res: tensor(199.3170) ,tv: tensor(214.1687) ,beta: 0.054999992
save epoch: 122 ,loss: tensor(191.8159) ,ic: tensor(0.8187) ,res: tensor(189.1007) ,tv: tensor(189.6478) ,beta: 0.05999999
[load model from: output/v0/e1_net_seq1.pth
best epoch:  47863 , min loss: 0.007478602696210146 , train time: 15054.726706981659

# train e1_net_seq2.pth
* training log:
[load model from: output/v0/p_net.pth
best epoch:  49652 , min loss: 0.002916318131610751 , train time: 6607.80632185936
[load model from: output/v0/e1_net_seq1.pth
best epoch:  47863 , min loss: 0.007478602696210146 , train time: 15054.726706981659
normalize:  0.017596237
save epoch: 0 ,loss: tensor(153.5240) ,ic: tensor(1.9939) ,res: tensor(150.7267) ,tv: tensor(80.3413) ,beta: 0.0
... RAR IC, add:  0.10000000149011612 5.809431552886963
... RAR Res, add:  0.10649897158145905 105.62612915039062
save epoch: 2 ,loss: tensor(145.3829) ,ic: tensor(2.8550) ,res: tensor(141.4457) ,tv: tensor(108.2193) ,beta: 0.005
save epoch: 3 ,loss: tensor(135.9893) ,ic: tensor(2.2954) ,res: tensor(132.7217) ,tv: tensor(97.2164) ,beta: 0.01
save epoch: 7 ,loss: tensor(127.9480) ,ic: tensor(1.2674) ,res: tensor(125.7385) ,tv: tensor(94.2126) ,beta: 0.015
save epoch: 12 ,loss: tensor(120.6564) ,ic: tensor(0.9650) ,res: tensor(118.6182) ,tv: tensor(107.3294) ,beta: 0.02
save epoch: 21 ,loss: tensor(114.4067) ,ic: tensor(0.5521) ,res: tensor(112.8783) ,tv: tensor(97.6325) ,beta: 0.024999999
save epoch: 30 ,loss: tensor(108.6225) ,ic: tensor(0.4111) ,res: tensor(107.1892) ,tv: tensor(102.2163) ,beta: 0.029999997
save epoch: 38 ,loss: tensor(102.7099) ,ic: tensor(0.3944) ,res: tensor(101.3302) ,tv: tensor(98.5264) ,beta: 0.034999996
save epoch: 45 ,loss: tensor(96.9351) ,ic: tensor(0.4154) ,res: tensor(95.6385) ,tv: tensor(88.1148) ,beta: 0.039999995
save epoch: 51 ,loss: tensor(92.0266) ,ic: tensor(0.4836) ,res: tensor(90.7086) ,tv: tensor(83.4446) ,beta: 0.044999994
save epoch: 57 ,loss: tensor(87.4047) ,ic: tensor(0.5182) ,res: tensor(85.9425) ,tv: tensor(94.4018) ,beta: 0.049999993
save epoch: 63 ,loss: tensor(82.4689) ,ic: tensor(0.4976) ,res: tensor(81.0081) ,tv: tensor(96.3139) ,beta: 0.054999992
save epoch: 68 ,loss: tensor(77.6828) ,ic: tensor(0.5182) ,res: tensor(76.3921) ,tv: tensor(77.2537) ,beta: 0.05999999
save epoch: 73 ,loss: tensor(73.0607) ,ic: tensor(0.5667) ,res: tensor(71.6994) ,tv: tensor(79.4637) ,beta: 0.06499999
save epoch: 77 ,loss: tensor(69.2982) ,ic: tensor(0.5909) ,res: tensor(67.9929) ,tv: tensor(71.4406) ,beta: 0.06999999
save epoch: 82 ,loss: tensor(65.1535) ,ic: tensor(0.6065) ,res: tensor(63.9170) ,tv: tensor(63.0002) ,beta: 0.074999996
save epoch: 87 ,loss: tensor(61.7417) ,ic: tensor(0.6329) ,res: tensor(60.4056) ,tv: tensor(70.3180) ,beta: 0.08
save epoch: 93 ,loss: tensor(58.4110) ,ic: tensor(0.6255) ,res: tensor(57.1886) ,tv: tensor(59.6901) ,beta: 0.085
... RAR IC, add:  0.10000000149011612 2.938400983810425
... RAR Res, add:  0.11155211180448532 102.19293975830078
save epoch: 44065 ,loss: tensor(0.0092) ,ic: tensor(0.0019) ,res: tensor(0.0053) ,tv: tensor(0.1975) ,beta: 0.75499946
... RAR IC, add:  0.10000000149011612 0.2379164695739746
... RAR Res, add:  0.19055086374282837 3.044058322906494
save epoch: 48011 ,loss: tensor(0.0086) ,ic: tensor(0.0018) ,res: tensor(0.0048) ,tv: tensor(0.1934) ,beta: 0.75999945
... RAR IC, add:  0.10000000149011612 0.2052624672651291
... RAR Res, add:  0.15603306889533997 1.4007089138031006
e1_net train complete
[load model from: output/v0/e1_net_seq2.pth
best epoch:  48011 , min loss: 0.008551913313567638 , train time: 15045.07952094078