# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np

cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
class NameRGB:
    def __init__(self,name,B,G,R):
        self.name = name
        self.B = B
        self.G = G
        self.R = R


def input_arg(argvs, argc):
    if (argc != 3):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s Input_filename Output_filename' % argvs[0]
        quit()        # プログラムの終了

    print 'Input filename = %s' % argvs[1]
    print 'Output filename = %s' % argvs[2]
    # 引数でとったディレクトリの文字列をリターン
    return argvs


def detect(frame):
    # メンバーの名前と矩形の色定義
    MemberList = []
    MemberList.append(NameRGB("Hyoyeon",117, 163, 27))
    MemberList.append(NameRGB("Jessica",253, 169, 27))
    MemberList.append(NameRGB("Seohyun",178, 25, 79))
    MemberList.append(NameRGB("Sooyoung",253, 217, 25))
    MemberList.append(NameRGB("Suuny",253, 26, 182))
    MemberList.append(NameRGB("Taeyeon",27, 58, 254))
    MemberList.append(NameRGB("Tiffany",195, 155, 244))
    MemberList.append(NameRGB("Yoona",26, 251, 253))
    MemberList.append(NameRGB("Yuri",27, 26, 253))

    #顔の認識
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 1,
                                     minSize = (100,100))
    for (x, y, w, h) in faces:
        image = frame[y:y+h, x:x+w]
        cv2.imwrite("face.png", image)
        image = caffe.io.load_image('face.png')
        predictions = classifier.predict([image], oversample=False)
        pred = np.argmax(predictions)

        for i, value in enumerate(MemberList):
            if pred == i+1:
                # 確率を代入，文字列変換
                probability = int(predictions[0][int(i+1)]*100)
                probability = str(probability) + "%"
                # 矩形設置
                cv2.rectangle(frame, (x, y), (x + w, y + h), (value.B, value.G, value.R), 2)
                cv2.putText(frame,value.name,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,(value.B, value.G, value.R),2,cv2.CV_AA)
                cv2.putText(frame,probability,(x + w - 120, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(value.B, value.G, value.R),2,cv2.CV_AA)
    return frame


if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数

    filepath = input_arg(argvs, argc)
    in_image = cv2.imread(filepath[1])
    out_image = filepath[2]
    mean_blob = caffe_pb2.BlobProto()
    with open('../snsd_mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(
    mean_blob.data,
    dtype=np.float32).reshape(
        (mean_blob.channels,
        mean_blob.height,
        mean_blob.width))
    classifier = caffe.Classifier(
        '../snsd_cifar10_quick.prototxt',
        '../snsd_cifar10_quick_150715_iter_4000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    frame = detect(in_image)
    cv2.imwrite(out_image, frame)
    if os.path.isfile("face.png"):
        os.remove("face.png")
