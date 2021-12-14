import numpy as np
import tensorflow as tf
import cv2
import time
import h5py
import OMCNN_2CLSTM as Network  # define the CNN
import random
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device -1 == CPU,  0 == GPU

batch_size = 8
jobID = -1
framesnum = 16
inputDim = 448
input_size = (inputDim, inputDim)
resize_shape = input_size
outputDim = 112
output_size = (outputDim, outputDim)
epoch_num = 15
overlapframe = 5  # 0~framesnum+frame_skip

random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5
dp_in = 1
dp_h = 1

targetname = 'LSTMconv_prefinal_loss05_dp075_075MC100-200000'
VideoName = 'LEDOV_human_talking.mp4'
CheckpointFile = '/path/to/DNN/DeePVS/model/pretrain/' + targetname


def _BatchExtraction(VideoCap, batchsize=batch_size, last_input=None, video_start=True):
    if video_start:
        _, frame = VideoCap.read()
        frame = cv2.resize(frame, resize_shape)
        frame = frame.astype(np.float32)
        frame = frame / 255.0 * 2 - 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Input_Batch = frame[np.newaxis, ...]
        for i in range(batchsize - 1):
            _, frame = VideoCap.read()
            frame = cv2.resize(frame, resize_shape)
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=0)
        Input_Batch = Input_Batch[np.newaxis, ...]
    else:
        Input_Batch = last_input[:, -overlapframe:, ...]
        for i in range(batchsize - overlapframe):
            _, frame = VideoCap.read()
            frame = cv2.resize(frame, resize_shape)
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[np.newaxis, np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=1)
    return Input_Batch


def main():
    net = Network.Net()
    net.is_training = False
    input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
    RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    net.inference(input, RNNmask_in, RNNmask_h)
    net.dp_in = dp_in
    net.dp_h = dp_h
    predicts = net.out

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.08
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, CheckpointFile)
    VideoName_short = VideoName[:-4]
    VideoCap = cv2.VideoCapture(VideoName)

    VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        'New video: %s with %d frames and size of (%d, %d)' % (VideoName_short, VideoFrame, VideoSize[1], VideoSize[0]))
    fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
    videoWriter = cv2.VideoWriter(VideoName_short + '_out.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                                  output_size, isColor=False)
    start_time = time.time()
    videostart = True
    Input_last = 0

    latencyFile = "{0}_runtime.txt".format(jobID)

    latencyResults = open(latencyFile, 'a')



    while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - (framesnum + frame_skip - overlapframe) * batch_size:
        if os.path.exists(str(jobID) + '_terminate.txt'):  # terminate the job
            break
        frameStart = time.time()
        for j in range(batch_size):
            if videostart:
                Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, video_start=videostart)
                videostart = False
                Input_last = Input_Batch
            else:
                Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, last_input=Input_last,video_start=videostart)
                Input_last = Input_Batch

            if j == 0:
                Input_Batch_Temp = Input_Batch
            else:
                Input_Batch_Temp = np.concatenate((Input_Batch_Temp, Input_Batch), axis=0)
        #print(np.shape(Input_Batch_Temp))

        mask_in = np.ones((batch_size, 28, 28, 128, 4 * 2))
        mask_h = np.ones((batch_size, 28, 28, 128, 4 * 2))
        np_predict = sess.run(predicts, feed_dict={input: Input_Batch_Temp, RNNmask_in: mask_in, RNNmask_h: mask_h})
        #print(np.shape(np_predict))

        for index1 in range(batch_size):
            for index2 in range(framesnum):
                Out_frame = np_predict[index1, index2, :, :, 0]
                Out_frame = Out_frame * 255
                Out_frame = np.uint8(Out_frame)
                videoWriter.write(Out_frame)
        frameEnd = time.time()
        latencyResults.write(str(frameEnd - frameStart) + "\n")
        latencyResults.flush()


    # restFrame = VideoFrame - VideoCap.get(cv2.CAP_PROP_POS_FRAMES)
    # overlap = framesnum - restFrame + overlapframe
    # Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, last_input=Input_last,video_start=False)
    # mask_in = np.ones((1, 28, 28, 128, 4 * 2))
    # mask_h = np.ones((1, 28, 28, 128, 4 * 2))
    # np_predict = sess.run(predicts, feed_dict={input: Input_Batch, RNNmask_in: mask_in, RNNmask_h: mask_h})
    # for index in range(restFrame):
    #	Out_frame = np_predict[0, framesnum - restFrame + index, :, :, 0]
    #	Out_frame = Out_frame * 255
    #	Out_frame = np.uint8(Out_frame)
    #	videoWriter.write(Out_frame)

    duration = float(time.time() - start_time)
    print('Total time for this video %f' % (duration))
    # print(duration)
    VideoCap.release()
    latencyResults.close()


if __name__ == '__main__':
    P = argparse.ArgumentParser(prog="test")
    P.add_argument('--batch_size', type=int, default=128)
    P.add_argument('--input_video', type=str, default="human_chatting06_out.avi")
    P.add_argument('--job_id', type=int, default=1)

    f, unparsed = P.parse_known_args()

    batch_size = f.batch_size
    jobID = f.job_id
    VideoName = f.input_video
    startTime = time.time()
    main()
    endTime = time.time()
    runtimeResults = open("Jobs_Total_Runtime.txt", 'a')
    runtimeResults.write(str(f.job_id) + "," + "DeePVS" + "," + str(endTime - startTime) + "\n")
    runtimeResults.close()
    finished = open(str(f.job_id) + "_finished.txt", 'w')
    finished.write('finished')
    finished.close()