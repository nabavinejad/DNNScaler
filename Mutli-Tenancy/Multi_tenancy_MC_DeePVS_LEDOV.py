import time
import csv
import operator
import os
import numpy as np
import argparse
import subprocess
import threading
from threading import Thread


maxMTL = 10 # Maximum number of co-located DNNs
MC_Estimations = []
N = 5
latencyReadings = [0] * N


class CountdownTask:

    def __init__(self):
        self._running = True


    def terminate(self):
        self._running = False


    def run(self):
        triggerLatencyProfile = 0
        time.sleep(3)
        while self._running:

            # if triggerLatencyProfile == 0:
            #     print("wait for 5 seconds \n\n\n\n")
            #     time.sleep(5)
            #     print("After wait for 5 seconds \n\n\n\n")
            #     triggerLatencyProfile = 1
            #os.system('nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv ,noheader,nounits, -f "Multi-tenancy_nvidiasmi_output.csv"')
            time.sleep(1)

            if os.path.exists('1_runtime.txt') == False:
                continue
            #if os.stat('1_runtime.txt').st_size <= (N + 3):
                #continue
            # read the last image inference time of the request
            with open('1_runtime.txt', 'r') as f:
                lines = f.read().splitlines()
                if len(lines) > N:
                    for i in range(1,N):
                        last_line = lines[-i]
                        lastResponseTime = float(last_line)
                        latencyReadings[i] = lastResponseTime * 1000
                f.close()



def readMatrixCompletion(DNN_type):

    with open('MatrixCompletion_ImageNet.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(row)
                line_count += 1
            else:
                if row[0] == DNN_type:
                    for i in range(maxMTL):
                        MC_Estimations.append(float(row[i+1]))
        print(MC_Estimations)



def main(result_DNNScaler, DNN, ExpectedLatency, biasStep):
    # Initial Deployment of all the request
    ExpectedLatency = int(ExpectedLatency)
    if os.path.exists(result_DNNScaler):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    runtimeResults = open(result_DNNScaler, append_write)

    #readMatrixCompletion(DNN)

    MC_Estimations = [1103.00, 1388.61, 1692.81, 2253.76, 2879.60, 3314.09, 3836.58, 4710.97, 4774.85, 5254.23]

    rememberLatency1 = 0
    rememberLatency2 = 0


    DNN_ID = -1
    DNNsDic = {}
    temp = -1
    for j in range(len(MC_Estimations)):
        if ExpectedLatency > MC_Estimations[j]:
            temp = j + 1


    for k in range(temp):
        DNN_ID = k + 1
        os.system("python DeePVS_DNNScaler.py --batch_size 1 --input_video \"LEDOV_human_talking.mp4\"  --job_id " + str(DNN_ID) + "  &")
        DNNsDic.update({DNN_ID:1})

        runtimeResults.write(str(time.time()) + ", Job Initialized , " + str(DNN_ID) + "\n")

        #time.sleep(1)

        timeWaitBias = biasStep

    time.sleep(biasStep*temp)

    while True:
        time.sleep(1)
        DoNotContinue = 0
        #check to see if the jobs are finished
        for tmp in DNNsDic.keys():
            if os.path.exists(str(tmp) + "_finished.txt") or os.path.exists(str(tmp) + "_terminate.txt"):
                DNNsDic[tmp] = 0

        AllFinished = 1
        for tmp in DNNsDic.keys():
            if DNNsDic[tmp] == 1:
                AllFinished = 0
        if AllFinished == 1:
            break


        averageLatency = np.mean(latencyReadings)

        #print(averageLatency)

        if averageLatency <= ExpectedLatency and averageLatency >= (0.85 * ExpectedLatency):
            DoNotContinue = 1
            pass    # Everythings OK, do nothing

        elif averageLatency < (0.85 * ExpectedLatency) and DNN_ID < maxMTL and DoNotContinue == 0:

            rememberLatency1 = averageLatency

            DNN_ID = DNN_ID + 1
            os.system("python DeePVS_DNNScaler.py --batch_size 1 --input_video \"LEDOV_human_talking.mp4\"  --job_id " + str(DNN_ID) + "  &")
            DNNsDic.update({DNN_ID:1})

            runtimeResults.write(str(time.time()) + ", Job Initialized , " + str(DNN_ID) + "\n")

            time.sleep(25 + timeWaitBias)
            timeWaitBias = timeWaitBias + biasStep

        elif averageLatency > ExpectedLatency and DoNotContinue == 0:

            rememberExpectedLatency = ExpectedLatency
            rememberLatency2 = averageLatency
            DNNsDic[DNN_ID] = 0
            runtimeResults.write(str(time.time()) + ", Job Terminated , " + str(DNN_ID) + "\n")
            finished = open(str(DNN_ID) + "_terminate.txt", 'w')
            finished.write(str(DNN_ID) + ' Job Terminated')

            DNNsDic.pop(DNN_ID)
            DNN_ID = DNN_ID - 1
            runtimeResults.write(str(time.time()) + ", Stable Number Was , " + str(DNN_ID) + "\n")
            finished.close()
            time.sleep(20)

            while (np.mean(latencyReadings) >= 0.95*rememberLatency1 and np.mean(latencyReadings) <= rememberLatency2) and (rememberExpectedLatency == ExpectedLatency):
                for tmp in DNNsDic.keys():
                    if os.path.exists(str(tmp) + "_finished.txt") or os.path.exists(str(tmp) + "_terminate.txt"):
                        DNNsDic[tmp] = 0

                AllFinished = 1
                for tmp in DNNsDic.keys():
                    if DNNsDic[tmp] == 1:
                        AllFinished = 0
                if AllFinished == 1:
                    break
                time.sleep(1)




    runtimeResults.close()





if __name__ == '__main__':
    P = argparse.ArgumentParser(prog="OurApproach")
    P.add_argument('--request_DNNScaler', type=str, default='input.txt')
    P.add_argument('--DNN', type=str, default='DNN_Name')
    P.add_argument('--expectedLatency', type=str, default='output.txt')
    P.add_argument('--biasWait', type=int, default=1)


    f,unparsed = P.parse_known_args()

    c = CountdownTask()
    t = Thread(target=c.run)
    t.start()
    os.system(
        'nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv ,nounits, -l 1 -f "MT_nvidiasmi_output.csv" &')

    main(f.request_DNNScaler, f.DNN, f.expectedLatency,f.biasWait)
    os.system('pkill nvidia-smi')
    c.terminate()
    t.join()

