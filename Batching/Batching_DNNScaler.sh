#!/bin/bash

python InceptionV4_BatchScaler.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 419  --result_file '04_InceptionV4_BatchScaler.txt' --topN 1

python NASNet_A_Large_331_BatchScaler.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 417  --result_file '10_NASNet_A_Large_331_BatchScaler.txt' --topN 1

python ResNetV2_101_BatchScaler.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 72  --result_file '15_ResNetV2_101_BatchScaler.txt' --topN 1

python ResNetV2_152_BatchScaler.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 206  --result_file '16_ResNetV2_152_BatchScaler.txt' --topN 1

python ResNetV2_101_BatchScaler.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 107  --result_file '19_ResNetV2_101_BatchScaler.txt' --topN 1



python InceptionV2_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 116  --result_file '22_InceptionV2_BatchScaler.txt' --topN 1

python InceptionV3_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 322  --result_file '23_InceptionV3_BatchScaler.txt' --topN 1

python InceptionV4_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 139  --result_file '24_InceptionV4_BatchScaler.txt' --topN 1

python PNASNet_5_Large_331_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 524  --result_file '32_PNASNet_5_Large_331_BatchScaler.txt' --topN 1

python PNASNet_5_Mobile_224_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 321  --result_file '33_PNASNet_5_Mobile_224_BatchScaler.txt' --topN 1

python ResNetV2_50_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 31  --result_file '34_ResNetV2_50_BatchScaler.txt' --topN 1

python ResNetV2_101_BatchScaler.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 107  --result_file '35_ResNetV2_101_BatchScaler.txt' --topN 1



exit 0