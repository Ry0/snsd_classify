# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np

cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

def input_arg(argvs, argc):
    if (argc != 3):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s srcdirectory outputdirectory' % argvs[0]
        quit()        # プログラムの終了

    print 'Input filename = %s' % argvs[1]
    print 'Output filename = %s' % argvs[2]
    # 引数でとったディレクトリの文字列をリターン
    return argvs


def detect(frame):
    #顔の認識
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 1,
                                     minSize = (150,150))
    for (x, y, w, h) in faces:
        image = frame[y:y+h, x:x+w]
        cv2.imwrite("face.png", image)
        image = caffe.io.load_image('face.png')
        predictions = classifier.predict([image], oversample=False)
        pred = np.argmax(predictions)
        if pred == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 79), 2)
            cv2.putText(frame,"ETC",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 150, 79),2)
        if pred == 1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (190, 165, 245), 2)
            cv2.putText(frame,"Hyoyeon",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(190, 165, 245),2)
        if pred == 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (147, 88, 120), 2)
            cv2.putText(frame,"Jessica",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(147, 88, 120),2)
        if pred == 3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (111, 180, 141), 2)
            cv2.putText(frame,"Seohyun",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(111, 180, 141),2)
        if pred == 4:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (161, 215, 244), 2)
            cv2.putText(frame,"Sooyoung",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(161, 215, 244),2)
        if pred == 5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 83, 191), 2)
            cv2.putText(frame,"Suuny",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 83, 191),2)
        if pred == 6:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (56, 215, 244), 2)
            cv2.putText(frame,"Taeyeon",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(56, 215, 244),2)
        if pred == 7:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (111, 255, 141), 2)
            cv2.putText(frame,"Tiffany",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(111, 255, 141),2)
        if pred == 8:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 87, 244), 2)
            cv2.putText(frame,"Yoona",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(80, 87, 244),2)
        if pred == 9:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (111, 255, 255), 2)
            cv2.putText(frame,"Yuri",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(111, 255, 255),2)
    return frame


if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数

    filepath = input_arg(argvs, argc)
    in_image = cv2.imread(filepath[1])
    out_image = filepath[2]
    mean_blob = caffe_pb2.BlobProto()
    with open('mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(
    mean_blob.data,
    dtype=np.float32).reshape(
        (mean_blob.channels,
        mean_blob.height,
        mean_blob.width))
    classifier = caffe.Classifier(
        'snsd_cifar10_quick.prototxt',
        'snsd_cifar10_quick_iter_4000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    frame = detect(in_image)
    cv2.imwrite(out_image, frame)
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         frame = detect(frame)
    #         out.write(frame)
    #     else:
    #         break
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
