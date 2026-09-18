
# Source-seed and split sensitivity

Each source-seed row summarizes five split seeds. Each split-seed row summarizes
five source seeds. The paired tables report within-unit differences before
grouping.

**Caveat on the A2 rows below:** the consistently high A2 Macro-F1/Accuracy
across every source seed and every split seed (all in the 0.96–0.99 range) is
expected under near-duplicate within-condition dependence and should not be
read as evidence of stable transferable learning — see the data-dependence
disclosure in `01_EXECUTIVE_SUMMARY.md` and `15_LIMITATIONS.md`. The A0 and A1
rows are the primary-endpoint sensitivity cuts and are not affected by this
caveat in the same way, since neither arm fits any parameter on P4.

## Arm metrics by source seed

| Arm | Metric | source_seed | n | Mean | Sample SD | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 matched source-only control | Accuracy | 42 | 5 | 0.174895 | 0.001214 | 0.174699 | 0.173633 | 0.176778 |
| A0 matched source-only control | Accuracy | 43 | 5 | 0.120457 | 0.001305 | 0.120516 | 0.118569 | 0.122137 |
| A0 matched source-only control | Accuracy | 44 | 5 | 0.129302 | 0.000762 | 0.129019 | 0.128617 | 0.130575 |
| A0 matched source-only control | Accuracy | 45 | 5 | 0.146431 | 0.002083 | 0.145382 | 0.144695 | 0.148730 |
| A0 matched source-only control | Accuracy | 46 | 5 | 0.115953 | 0.001715 | 0.115276 | 0.114550 | 0.118923 |
| A0 matched source-only control | Macro-F1 | 42 | 5 | 0.124724 | 0.000809 | 0.124450 | 0.124069 | 0.126135 |
| A0 matched source-only control | Macro-F1 | 43 | 5 | 0.097663 | 0.002212 | 0.097204 | 0.094652 | 0.100434 |
| A0 matched source-only control | Macro-F1 | 44 | 5 | 0.101198 | 0.000852 | 0.101011 | 0.100090 | 0.102312 |
| A0 matched source-only control | Macro-F1 | 45 | 5 | 0.133972 | 0.002209 | 0.132918 | 0.131474 | 0.136459 |
| A0 matched source-only control | Macro-F1 | 46 | 5 | 0.099629 | 0.001649 | 0.098906 | 0.098417 | 0.102490 |
| A1 matched target-informed selection | Accuracy | 42 | 5 | 0.157917 | 0.023727 | 0.173633 | 0.130924 | 0.176778 |
| A1 matched target-informed selection | Accuracy | 43 | 5 | 0.122787 | 0.001520 | 0.123392 | 0.120516 | 0.124498 |
| A1 matched target-informed selection | Accuracy | 44 | 5 | 0.138877 | 0.006241 | 0.139413 | 0.128916 | 0.145909 |
| A1 matched target-informed selection | Accuracy | 45 | 5 | 0.192673 | 0.024399 | 0.203376 | 0.149056 | 0.204819 |
| A1 matched target-informed selection | Accuracy | 46 | 5 | 0.128894 | 0.017115 | 0.121322 | 0.120579 | 0.159502 |
| A1 matched target-informed selection | Macro-F1 | 42 | 5 | 0.122678 | 0.003247 | 0.124069 | 0.118260 | 0.126135 |
| A1 matched target-informed selection | Macro-F1 | 43 | 5 | 0.097327 | 0.001180 | 0.097015 | 0.095906 | 0.099104 |
| A1 matched target-informed selection | Macro-F1 | 44 | 5 | 0.101640 | 0.001160 | 0.101310 | 0.100851 | 0.103658 |
| A1 matched target-informed selection | Macro-F1 | 45 | 5 | 0.142696 | 0.003800 | 0.143706 | 0.136091 | 0.145828 |
| A1 matched target-informed selection | Macro-F1 | 46 | 5 | 0.104368 | 0.003781 | 0.105639 | 0.097708 | 0.106951 |
| A2 final selected adaptation | Accuracy | 42 | 5 | 0.986895 | 0.003164 | 0.986736 | 0.982315 | 0.990730 |
| A2 final selected adaptation | Accuracy | 43 | 5 | 0.982226 | 0.004025 | 0.981526 | 0.976622 | 0.986334 |
| A2 final selected adaptation | Accuracy | 44 | 5 | 0.982630 | 0.002883 | 0.981519 | 0.980305 | 0.987540 |
| A2 final selected adaptation | Accuracy | 45 | 5 | 0.980788 | 0.010636 | 0.984727 | 0.961847 | 0.986699 |
| A2 final selected adaptation | Accuracy | 46 | 5 | 0.984962 | 0.001947 | 0.984739 | 0.982724 | 0.987138 |
| A2 final selected adaptation | Macro-F1 | 42 | 5 | 0.986915 | 0.003173 | 0.986768 | 0.982310 | 0.990751 |
| A2 final selected adaptation | Macro-F1 | 43 | 5 | 0.982209 | 0.004048 | 0.981522 | 0.976591 | 0.986365 |
| A2 final selected adaptation | Macro-F1 | 44 | 5 | 0.982591 | 0.002910 | 0.981420 | 0.980278 | 0.987552 |
| A2 final selected adaptation | Macro-F1 | 45 | 5 | 0.980798 | 0.010588 | 0.984719 | 0.961943 | 0.986696 |
| A2 final selected adaptation | Macro-F1 | 46 | 5 | 0.984955 | 0.001984 | 0.984723 | 0.982674 | 0.987189 |
| A2 linear probe (selected configuration) | Accuracy | 42 | 5 | 0.986895 | 0.003164 | 0.986736 | 0.982315 | 0.990730 |
| A2 linear probe (selected configuration) | Accuracy | 43 | 5 | 0.982226 | 0.004025 | 0.981526 | 0.976622 | 0.986334 |
| A2 linear probe (selected configuration) | Accuracy | 44 | 5 | 0.982630 | 0.002883 | 0.981519 | 0.980305 | 0.987540 |
| A2 linear probe (selected configuration) | Accuracy | 45 | 5 | 0.980788 | 0.010636 | 0.984727 | 0.961847 | 0.986699 |
| A2 linear probe (selected configuration) | Accuracy | 46 | 5 | 0.984962 | 0.001947 | 0.984739 | 0.982724 | 0.987138 |
| A2 linear probe (selected configuration) | Macro-F1 | 42 | 5 | 0.986915 | 0.003173 | 0.986768 | 0.982310 | 0.990751 |
| A2 linear probe (selected configuration) | Macro-F1 | 43 | 5 | 0.982209 | 0.004048 | 0.981522 | 0.976591 | 0.986365 |
| A2 linear probe (selected configuration) | Macro-F1 | 44 | 5 | 0.982591 | 0.002910 | 0.981420 | 0.980278 | 0.987552 |
| A2 linear probe (selected configuration) | Macro-F1 | 45 | 5 | 0.980798 | 0.010588 | 0.984719 | 0.961943 | 0.986696 |
| A2 linear probe (selected configuration) | Macro-F1 | 46 | 5 | 0.984955 | 0.001984 | 0.984723 | 0.982674 | 0.987189 |
| A2 prototype blend (selected configuration) | Accuracy | 42 | 5 | 0.440442 | 0.033306 | 0.450161 | 0.396386 | 0.478436 |
| A2 prototype blend (selected configuration) | Accuracy | 43 | 5 | 0.437522 | 0.031396 | 0.430695 | 0.410040 | 0.491559 |
| A2 prototype blend (selected configuration) | Accuracy | 44 | 5 | 0.398684 | 0.024353 | 0.401451 | 0.375502 | 0.434713 |
| A2 prototype blend (selected configuration) | Accuracy | 45 | 5 | 0.448709 | 0.030696 | 0.457395 | 0.398795 | 0.481720 |
| A2 prototype blend (selected configuration) | Accuracy | 46 | 5 | 0.362825 | 0.020541 | 0.372186 | 0.329984 | 0.380072 |
| A2 prototype blend (selected configuration) | Macro-F1 | 42 | 5 | 0.414731 | 0.040911 | 0.417877 | 0.366879 | 0.469149 |
| A2 prototype blend (selected configuration) | Macro-F1 | 43 | 5 | 0.427067 | 0.029782 | 0.415260 | 0.404998 | 0.479166 |
| A2 prototype blend (selected configuration) | Macro-F1 | 44 | 5 | 0.394376 | 0.022714 | 0.398015 | 0.368541 | 0.427475 |
| A2 prototype blend (selected configuration) | Macro-F1 | 45 | 5 | 0.437476 | 0.029874 | 0.444781 | 0.391266 | 0.471850 |
| A2 prototype blend (selected configuration) | Macro-F1 | 46 | 5 | 0.340098 | 0.027070 | 0.352039 | 0.296442 | 0.362268 |

## Paired effects by source seed

| Comparison | Metric | source_seed | n | Mean Δ | Sample SD | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 − A0 | Accuracy | 42 | 5 | -0.016978 | 0.023266 | +0.000000 | -0.043775 | +0.000000 |
| A1 − A0 | Accuracy | 43 | 5 | +0.002330 | 0.002315 | +0.002412 | +0.000000 | +0.004823 |
| A1 − A0 | Accuracy | 44 | 5 | +0.009575 | 0.006105 | +0.010048 | +0.000000 | +0.016526 |
| A1 − A0 | Accuracy | 45 | 5 | +0.046242 | 0.025736 | +0.058682 | +0.000402 | +0.059438 |
| A1 − A0 | Accuracy | 46 | 5 | +0.012941 | 0.015450 | +0.006029 | +0.006024 | +0.040579 |
| A1 − A0 | Macro-F1 | 42 | 5 | -0.002046 | 0.002903 | +0.000000 | -0.006190 | +0.000000 |
| A1 − A0 | Macro-F1 | 43 | 5 | -0.000337 | 0.001798 | +0.000000 | -0.003419 | +0.001253 |
| A1 − A0 | Macro-F1 | 44 | 5 | +0.000443 | 0.001197 | +0.000000 | -0.001002 | +0.001934 |
| A1 − A0 | Macro-F1 | 45 | 5 | +0.008724 | 0.005408 | +0.011436 | -0.000059 | +0.012911 |
| A1 − A0 | Macro-F1 | 46 | 5 | +0.004739 | 0.005349 | +0.006805 | -0.004781 | +0.008045 |
| A2 final − A0 | Accuracy | 42 | 5 | +0.812000 | 0.002875 | +0.811495 | +0.808682 | +0.816606 |
| A2 final − A0 | Accuracy | 43 | 5 | +0.861769 | 0.004702 | +0.861446 | +0.856106 | +0.867765 |
| A2 final − A0 | Accuracy | 44 | 5 | +0.853328 | 0.003322 | +0.851673 | +0.850944 | +0.858923 |
| A2 final − A0 | Accuracy | 45 | 5 | +0.834357 | 0.010053 | +0.837969 | +0.816466 | +0.840032 |
| A2 final − A0 | Accuracy | 46 | 5 | +0.869009 | 0.003307 | +0.869478 | +0.863801 | +0.872186 |
| A2 final − A0 | Macro-F1 | 42 | 5 | +0.862190 | 0.002888 | +0.862222 | +0.858242 | +0.866328 |
| A2 final − A0 | Macro-F1 | 43 | 5 | +0.884545 | 0.004614 | +0.884318 | +0.879668 | +0.891713 |
| A2 final − A0 | Macro-F1 | 44 | 5 | +0.881393 | 0.003076 | +0.880188 | +0.879108 | +0.886541 |
| A2 final − A0 | Macro-F1 | 45 | 5 | +0.846826 | 0.010029 | +0.850237 | +0.829026 | +0.853246 |
| A2 final − A0 | Macro-F1 | 46 | 5 | +0.885326 | 0.003175 | +0.886306 | +0.880185 | +0.887956 |
| A2 final − A1 | Accuracy | 42 | 5 | +0.828978 | 0.025055 | +0.811973 | +0.808682 | +0.857719 |
| A2 final − A1 | Accuracy | 43 | 5 | +0.859439 | 0.003144 | +0.858578 | +0.856106 | +0.862942 |
| A2 final − A1 | Accuracy | 44 | 5 | +0.843754 | 0.007559 | +0.842105 | +0.835147 | +0.853815 |
| A2 final − A1 | Accuracy | 45 | 5 | +0.788114 | 0.029585 | +0.781350 | +0.757028 | +0.837284 |
| A2 final − A1 | Accuracy | 46 | 5 | +0.856068 | 0.018429 | +0.863454 | +0.823222 | +0.866158 |
| A2 final − A1 | Macro-F1 | 42 | 5 | +0.864237 | 0.004802 | +0.862638 | +0.858242 | +0.870369 |
| A2 final − A1 | Macro-F1 | 43 | 5 | +0.884882 | 0.004657 | +0.883836 | +0.879668 | +0.890460 |
| A2 final − A1 | Macro-F1 | 44 | 5 | +0.880951 | 0.003596 | +0.880110 | +0.877344 | +0.886656 |
| A2 final − A1 | Macro-F1 | 45 | 5 | +0.838102 | 0.012920 | +0.841013 | +0.816115 | +0.850215 |
| A2 final − A1 | Macro-F1 | 46 | 5 | +0.880588 | 0.003061 | +0.880869 | +0.876491 | +0.884966 |

## Arm metrics by split seed

| Arm | Metric | p4_split_seed | n | Mean | Sample SD | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 matched source-only control | Accuracy | 11 | 5 | 0.136495 | 0.023616 | 0.128617 | 0.114550 | 0.173633 |
| A0 matched source-only control | Accuracy | 23 | 5 | 0.136867 | 0.024060 | 0.128916 | 0.115261 | 0.174699 |
| A0 matched source-only control | Accuracy | 37 | 5 | 0.139413 | 0.023864 | 0.130575 | 0.118923 | 0.176778 |
| A0 matched source-only control | Accuracy | 53 | 5 | 0.136656 | 0.024371 | 0.129019 | 0.115756 | 0.175241 |
| A0 matched source-only control | Accuracy | 71 | 5 | 0.137606 | 0.024060 | 0.129383 | 0.115276 | 0.174123 |
| A0 matched source-only control | Macro-F1 | 11 | 5 | 0.111433 | 0.015875 | 0.101011 | 0.098792 | 0.132859 |
| A0 matched source-only control | Macro-F1 | 23 | 5 | 0.110768 | 0.016678 | 0.100851 | 0.097204 | 0.132918 |
| A0 matched source-only control | Macro-F1 | 37 | 5 | 0.113238 | 0.016778 | 0.102490 | 0.099104 | 0.136150 |
| A0 matched source-only control | Macro-F1 | 53 | 5 | 0.110061 | 0.016702 | 0.100090 | 0.094652 | 0.131474 |
| A0 matched source-only control | Macro-F1 | 71 | 5 | 0.111687 | 0.017723 | 0.101724 | 0.096924 | 0.136459 |
| A1 matched target-informed selection | Accuracy | 11 | 5 | 0.152010 | 0.035821 | 0.138666 | 0.120579 | 0.203778 |
| A1 matched target-informed selection | Accuracy | 23 | 5 | 0.142088 | 0.035269 | 0.128916 | 0.121285 | 0.204819 |
| A1 matched target-informed selection | Accuracy | 37 | 5 | 0.149377 | 0.020583 | 0.149056 | 0.122137 | 0.176778 |
| A1 matched target-informed selection | Accuracy | 53 | 5 | 0.153055 | 0.035412 | 0.141479 | 0.121785 | 0.203376 |
| A1 matched target-informed selection | Accuracy | 71 | 5 | 0.144619 | 0.033881 | 0.133011 | 0.120516 | 0.202338 |
| A1 matched target-informed selection | Macro-F1 | 11 | 5 | 0.114383 | 0.019676 | 0.105639 | 0.097015 | 0.144295 |
| A1 matched target-informed selection | Macro-F1 | 23 | 5 | 0.113569 | 0.019662 | 0.105221 | 0.097686 | 0.145828 |
| A1 matched target-informed selection | Macro-F1 | 37 | 5 | 0.112070 | 0.017783 | 0.101310 | 0.097708 | 0.136091 |
| A1 matched target-informed selection | Macro-F1 | 53 | 5 | 0.114393 | 0.019595 | 0.106320 | 0.095906 | 0.143706 |
| A1 matched target-informed selection | Macro-F1 | 71 | 5 | 0.114295 | 0.018454 | 0.106951 | 0.096924 | 0.143559 |
| A2 final selected adaptation | Accuracy | 11 | 5 | 0.985370 | 0.002081 | 0.985932 | 0.982315 | 0.987540 |
| A2 final selected adaptation | Accuracy | 23 | 5 | 0.979357 | 0.009938 | 0.982731 | 0.961847 | 0.985944 |
| A2 final selected adaptation | Accuracy | 37 | 5 | 0.984010 | 0.003414 | 0.982724 | 0.980715 | 0.988751 |
| A2 final selected adaptation | Accuracy | 53 | 5 | 0.985048 | 0.002805 | 0.986334 | 0.980305 | 0.987138 |
| A2 final selected adaptation | Accuracy | 71 | 5 | 0.983716 | 0.005375 | 0.983474 | 0.976622 | 0.990730 |
| A2 final selected adaptation | Macro-F1 | 11 | 5 | 0.985372 | 0.002087 | 0.985921 | 0.982310 | 0.987552 |
| A2 final selected adaptation | Macro-F1 | 23 | 5 | 0.979372 | 0.009895 | 0.982703 | 0.961943 | 0.985971 |
| A2 final selected adaptation | Macro-F1 | 37 | 5 | 0.983964 | 0.003456 | 0.982674 | 0.980644 | 0.988773 |
| A2 final selected adaptation | Macro-F1 | 53 | 5 | 0.985064 | 0.002834 | 0.986365 | 0.980278 | 0.987189 |
| A2 final selected adaptation | Macro-F1 | 71 | 5 | 0.983697 | 0.005398 | 0.983442 | 0.976591 | 0.990751 |
| A2 linear probe (selected configuration) | Accuracy | 11 | 5 | 0.985370 | 0.002081 | 0.985932 | 0.982315 | 0.987540 |
| A2 linear probe (selected configuration) | Accuracy | 23 | 5 | 0.979357 | 0.009938 | 0.982731 | 0.961847 | 0.985944 |
| A2 linear probe (selected configuration) | Accuracy | 37 | 5 | 0.984010 | 0.003414 | 0.982724 | 0.980715 | 0.988751 |
| A2 linear probe (selected configuration) | Accuracy | 53 | 5 | 0.985048 | 0.002805 | 0.986334 | 0.980305 | 0.987138 |
| A2 linear probe (selected configuration) | Accuracy | 71 | 5 | 0.983716 | 0.005375 | 0.983474 | 0.976622 | 0.990730 |
| A2 linear probe (selected configuration) | Macro-F1 | 11 | 5 | 0.985372 | 0.002087 | 0.985921 | 0.982310 | 0.987552 |
| A2 linear probe (selected configuration) | Macro-F1 | 23 | 5 | 0.979372 | 0.009895 | 0.982703 | 0.961943 | 0.985971 |
| A2 linear probe (selected configuration) | Macro-F1 | 37 | 5 | 0.983964 | 0.003456 | 0.982674 | 0.980644 | 0.988773 |
| A2 linear probe (selected configuration) | Macro-F1 | 53 | 5 | 0.985064 | 0.002834 | 0.986365 | 0.980278 | 0.987189 |
| A2 linear probe (selected configuration) | Macro-F1 | 71 | 5 | 0.983697 | 0.005398 | 0.983442 | 0.976591 | 0.990751 |
| A2 prototype blend (selected configuration) | Accuracy | 11 | 5 | 0.433119 | 0.045775 | 0.446543 | 0.372186 | 0.491559 |
| A2 prototype blend (selected configuration) | Accuracy | 23 | 5 | 0.387309 | 0.021574 | 0.396386 | 0.355823 | 0.410040 |
| A2 prototype blend (selected configuration) | Accuracy | 37 | 5 | 0.437525 | 0.038187 | 0.434713 | 0.380072 | 0.481720 |
| A2 prototype blend (selected configuration) | Accuracy | 53 | 5 | 0.400965 | 0.048987 | 0.416801 | 0.329984 | 0.457395 |
| A2 prototype blend (selected configuration) | Accuracy | 71 | 5 | 0.429262 | 0.041581 | 0.431278 | 0.376058 | 0.478436 |
| A2 prototype blend (selected configuration) | Macro-F1 | 11 | 5 | 0.420283 | 0.045245 | 0.430011 | 0.358018 | 0.479166 |
| A2 prototype blend (selected configuration) | Macro-F1 | 23 | 5 | 0.372682 | 0.027919 | 0.368541 | 0.331726 | 0.404998 |
| A2 prototype blend (selected configuration) | Macro-F1 | 37 | 5 | 0.418946 | 0.039047 | 0.417877 | 0.362268 | 0.471850 |
| A2 prototype blend (selected configuration) | Macro-F1 | 53 | 5 | 0.383287 | 0.055356 | 0.383545 | 0.296442 | 0.444781 |
| A2 prototype blend (selected configuration) | Macro-F1 | 71 | 5 | 0.418551 | 0.045592 | 0.422491 | 0.352039 | 0.469149 |

## Paired effects by split seed

| Comparison | Metric | p4_split_seed | n | Mean Δ | Sample SD | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 − A0 | Accuracy | 11 | 5 | +0.015514 | 0.024650 | +0.006029 | +0.000000 | +0.059084 |
| A1 − A0 | Accuracy | 23 | 5 | +0.005221 | 0.036635 | +0.004418 | -0.043775 | +0.059438 |
| A1 − A0 | Accuracy | 37 | 5 | +0.009964 | 0.017525 | +0.000402 | +0.000000 | +0.040579 |
| A1 − A0 | Accuracy | 53 | 5 | +0.016399 | 0.024051 | +0.006029 | +0.000000 | +0.058682 |
| A1 − A0 | Accuracy | 71 | 5 | +0.007013 | 0.034014 | +0.006046 | -0.041112 | +0.053607 |
| A1 − A0 | Macro-F1 | 11 | 5 | +0.002950 | 0.006039 | +0.000000 | -0.003419 | +0.011436 |
| A1 − A0 | Macro-F1 | 23 | 5 | +0.002802 | 0.007285 | +0.000482 | -0.006190 | +0.012911 |
| A1 − A0 | Macro-F1 | 37 | 5 | -0.001168 | 0.002064 | -0.000059 | -0.004781 | +0.000000 |
| A1 − A0 | Macro-F1 | 53 | 5 | +0.004332 | 0.005130 | +0.001397 | +0.000000 | +0.012232 |
| A1 − A0 | Macro-F1 | 71 | 5 | +0.002608 | 0.005030 | +0.001934 | -0.004041 | +0.008045 |
| A2 final − A0 | Accuracy | 11 | 5 | +0.848875 | 0.025514 | +0.858923 | +0.808682 | +0.872186 |
| A2 final − A0 | Accuracy | 23 | 5 | +0.842490 | 0.026784 | +0.853815 | +0.811245 | +0.869478 |
| A2 final − A0 | Accuracy | 37 | 5 | +0.844596 | 0.020716 | +0.850944 | +0.811973 | +0.863801 |
| A2 final − A0 | Accuracy | 53 | 5 | +0.848392 | 0.024205 | +0.851286 | +0.811495 | +0.871383 |
| A2 final − A0 | Accuracy | 71 | 5 | +0.846110 | 0.019720 | +0.851673 | +0.816606 | +0.868198 |
| A2 final − A0 | Macro-F1 | 11 | 5 | +0.873939 | 0.017607 | +0.885488 | +0.851467 | +0.887956 |
| A2 final − A0 | Macro-F1 | 23 | 5 | +0.868605 | 0.024251 | +0.881852 | +0.829026 | +0.886306 |
| A2 final − A0 | Macro-F1 | 37 | 5 | +0.870725 | 0.013831 | +0.879108 | +0.850157 | +0.881540 |
| A2 final − A0 | Macro-F1 | 53 | 5 | +0.875003 | 0.016604 | +0.880188 | +0.853246 | +0.891713 |
| A2 final − A0 | Macro-F1 | 71 | 5 | +0.872010 | 0.013917 | +0.879278 | +0.850237 | +0.884537 |
| A2 final − A1 | Accuracy | 11 | 5 | +0.833360 | 0.037303 | +0.848875 | +0.780547 | +0.866158 |
| A2 final − A1 | Accuracy | 23 | 5 | +0.837269 | 0.045010 | +0.855020 | +0.757028 | +0.863454 |
| A2 final − A1 | Accuracy | 37 | 5 | +0.834632 | 0.017888 | +0.837284 | +0.811973 | +0.858578 |
| A2 final − A1 | Accuracy | 53 | 5 | +0.831994 | 0.035715 | +0.838826 | +0.781350 | +0.865354 |
| A2 final − A1 | Accuracy | 71 | 5 | +0.839097 | 0.032324 | +0.856106 | +0.784361 | +0.862152 |
| A2 final − A1 | Macro-F1 | 11 | 5 | +0.870989 | 0.021153 | +0.881109 | +0.840031 | +0.888907 |
| A2 final − A1 | Macro-F1 | 23 | 5 | +0.865803 | 0.028473 | +0.879502 | +0.816115 | +0.883836 |
| A2 final − A1 | Macro-F1 | 37 | 5 | +0.871894 | 0.014892 | +0.880110 | +0.850215 | +0.884966 |
| A2 final − A1 | Macro-F1 | 53 | 5 | +0.870671 | 0.019445 | +0.878791 | +0.841013 | +0.890460 |
| A2 final − A1 | Macro-F1 | 71 | 5 | +0.869402 | 0.015079 | +0.876491 | +0.843137 | +0.879668 |

These cuts are part of the preregistered consistency assessment. They are not
separate confirmatory experiments.
