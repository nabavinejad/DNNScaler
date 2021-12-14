#!/bin/bash

python InceptionV1_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 35  --result_file '01_InceptionV1_Clipper.txt' --topN 1
 
python InceptionV2_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 53  --result_file '02_InceptionV2_Clipper.txt' --topN 1
  
python InceptionV4_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 419  --result_file '04_InceptionV4_Clipper.txt' --topN 1

python MobilenetV1_05_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 199  --result_file '06_MobilenetV1_05_Clipper.txt' --topN 1

python MobilenetV1_025_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 186  --result_file '07_MobilenetV1_025_Clipper.txt' --topN 1

python MobilenetV2_1_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 81  --result_file '08_MobilenetV2_1_Clipper.txt' --topN 1

python NASNet_A_Large_331_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 417  --result_file '10_NASNet_A_Large_331_Clipper.txt' --topN 1

python NASNet_A_Mobile_224_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 85  --result_file '11_NASNet_A_Mobile_224_Clipper.txt' --topN 1

python PNASNet_5_Mobile_224_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 82  --result_file '13_PNASNet_5_Mobile_224_Clipper.txt' --topN 1

python ResNetV2_50_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 45  --result_file '14_ResNetV2_50_Clipper.txt' --topN 1

python ResNetV2_101_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 72  --result_file '15_ResNetV2_101_Clipper.txt' --topN 1

python ResNetV2_152_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 206  --result_file '16_ResNetV2_152_Clipper.txt' --topN 1

python ResNetV2_101_Clipper.py  --native --batch_size 1  --image_folder 'ILSVRC2012_15000' --latency 107  --result_file '19_ResNetV2_101_Clipper.txt' --topN 1




python InceptionV1_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 48  --result_file '21_InceptionV1_Clipper.txt' --topN 1

python InceptionV2_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 116  --result_file '22_InceptionV2_Clipper.txt' --topN 1

python InceptionV3_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 322  --result_file '23_InceptionV3_Clipper.txt' --topN 1

python InceptionV4_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 139  --result_file '24_InceptionV4_Clipper.txt' --topN 1

python MobilenetV1_1_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 89  --result_file '25_MobilenetV1_1_Clipper.txt' --topN 1

python MobilenetV1_05_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 60  --result_file '26_MobilenetV1_05_Clipper.txt' --topN 1

python MobilenetV1_025_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 104  --result_file '27_MobilenetV1_025_Clipper.txt' --topN 1

python MobilenetV2_1_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 129  --result_file '28_MobilenetV2_1_Clipper.txt' --topN 1

python PNASNet_5_Large_331_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 524  --result_file '32_PNASNet_5_Large_331_Clipper.txt' --topN 1

python PNASNet_5_Mobile_224_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 321  --result_file '33_PNASNet_5_Mobile_224_Clipper.txt' --topN 1

python ResNetV2_50_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 31  --result_file '34_ResNetV2_50_Clipper.txt' --topN 1

python ResNetV2_101_Clipper.py  --native --batch_size 1  --image_folder 'Caltech_256_20000' --latency 107  --result_file '35_ResNetV2_101_Clipper.txt' --topN 1


exit 0