python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_InceptionV1.py --expectedLatency 35   --biasWait 3   # Job 1 

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_InceptionV2.py --expectedLatency 53  --biasWait 3  # Job 2

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV1_05.py --expectedLatency 199  --biasWait 1     # Job 4  

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV1_025.py --expectedLatency 186  --biasWait 1    # Job 5  

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV2_1.py --expectedLatency 81  --biasWait 1    # Job 6 

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_NASNet_A_Mobile_224.py --expectedLatency 85  --biasWait 3    # Job 8

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_PNASNet_5_Mobile_224.py --expectedLatency 82  --biasWait 3    # Job 9

python Multi_tenancy_MC.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_ResNetV2_50.py --expectedLatency 45   --biasWait 3   # Job 10



python Multi_tenancy_MC_Caltech.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_InceptionV1.py --expectedLatency 48  --biasWait 3    # Job 14

python Multi_tenancy_MC_Caltech.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV1_1.py --expectedLatency 89  --biasWait 1    # Job 18

python Multi_tenancy_MC_Caltech.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV1_05.py --expectedLatency 60 --biasWait 1     # Job 19

python Multi_tenancy_MC_Caltech.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV1_025.py --expectedLatency 104  --biasWait 1    # Job 20

python Multi_tenancy_MC_Caltech.py --request_DNNScaler MT_DnnSclaer.txt --DNN MT_MobilenetV2_1.py --expectedLatency 129 --biasWait 1     # Job 21


python Multi_tenancy_MC_DeePVS_LEDOV.py --request_DNNScaler DeePVS_LEDOV.txt  --expectedLatency 3000 --biasWait 15     # Job 29

python Multi_tenancy_MC_DeePVS_DHF1K.py --request_DNNScaler DeePVS_DHF1K.txt  --expectedLatency 5000 --biasWait 15     # Job 30

