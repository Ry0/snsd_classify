# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from datetime import datetime

cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")


class NameRGB:
    def __init__(self,name,B,G,R):
        self.name = name
        self.B = B
        self.G = G
        self.R = R


def input_arg(argvs, argc):
    if (argc < 2 or 3 < argc):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s Input_filename Output_filename' % argvs[0]
        quit()        # プログラムの終了
    elif (argc == 2):
        d = datetime.now()
        filename = "img-" + d.strftime('%Y') + "-" + d.strftime('%m') + "-" + d.strftime('%d') + "-" + d.strftime('%H') + "h-" + d.strftime('%M') + "m" + d.strftime('%S')  + "s.jpg"
        argvs.append(filename)

    print 'Input filename = %s' % argvs[1]
    print 'Output filename = %s' % argvs[2]
    # 引数でとったディレクトリの文字列をリターン
    return argvs


def create_directory(output_directory, out_image_filename):
    if os.path.isdir(output_directory) == 0:
        print "Not exist \"%s\" folder. So create it." % output_directory
        os.makedirs(output_directory)
    else:
        print "Exist \"%s\" folder." % output_directory
    out_image = output_directory + "/" + out_image_filename
    return out_image


def detect(frame):
    # メンバーの名前と矩形の色定義
    MemberList = []
    MemberList.append(NameRGB("ETC",117, 163, 27))
    MemberList.append(NameRGB("Hyoyeon",117, 163, 27))
    MemberList.append(NameRGB("Jessica",253, 169, 27))
    MemberList.append(NameRGB("Seohyun",178, 25, 79))
    MemberList.append(NameRGB("Sooyoung",253, 217, 25))
    MemberList.append(NameRGB("Sunny",253, 26, 182))
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
                                     minSize = (75,75))
    for (x, y, w, h) in faces:
        image = frame[y:y+h, x:x+w]
        cv2.imwrite("face.png", image)
        os.system("convert face.png -equalize face.png")
        image = caffe.io.load_image('face.png')
        predictions = classifier.predict([image], oversample=False)
        pred = np.argmax(predictions)
        sorted_prediction_ind = sorted(range(len(predictions[0])),key=lambda x:predictions[0][x],reverse=True)
        first = MemberList[sorted_prediction_ind[0]].name + " " + str(int(predictions[0,sorted_prediction_ind[0]]*100)) + "%"
        second = MemberList[sorted_prediction_ind[1]].name + " " + str(round(predictions[0,sorted_prediction_ind[1]]*100,1)) + "%"
        third = MemberList[sorted_prediction_ind[2]].name + " " + str(round(predictions[0,sorted_prediction_ind[2]]*100,1)) + "%"

        for i, value in enumerate(MemberList):
            if pred == 0:
                print "Skip ETC!"
            elif pred == i:
                # 矩形設置
                cv2.rectangle(frame, (x, y), (x + w, y + h), (value.B, value.G, value.R), 2)
                cv2.putText(frame,first,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,(value.B, value.G, value.R),2,cv2.CV_AA)
                cv2.putText(frame,second,(x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(value.B, value.G, value.R),2,cv2.CV_AA)
                cv2.putText(frame,third,(x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(value.B, value.G, value.R),2,cv2.CV_AA)
    if os.path.isfile("face.png"):
        os.remove("face.png")
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
        '../snsd_cifar10_full.prototxt',
        '../snsd_cifar10_full_150717_iter_60000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    frame = detect(in_image)

    cv2.imshow("Show Image",frame)
    # キー入力待機
    key = cv2.waitKey(0)
    # ウィンドウ破棄
    cv2.destroyAllWindows()

    if key == 1048691:
        #保存先ディレクトリと保存名を指定
        out_image = create_directory("../success_img", out_image)
        cv2.imwrite(out_image, frame)
        print "Save image -> " + out_image
