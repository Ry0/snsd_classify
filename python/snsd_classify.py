# -*- coding:utf-8 -*-
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import sys

def input_arg(argvs, argc):
    if (argc != 2):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s srcdirectory outputdirectory' % argvs[0]
        quit()        # プログラムの終了

    print 'Filename = %s' % argvs[1]
    # 引数でとったディレクトリの文字列をリターン
    return argvs


if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数
    image_path = input_arg(argvs, argc)
    image_src = image_path[1]

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

    image = caffe.io.load_image(image_src)
    predictions = classifier.predict([image], oversample=False)
    pred = np.argmax(predictions)
    print(predictions)
    print(pred)
    name = ["etc","hyoyeon","jessica","seohyun","sooyoung","sunny","taeyeon","tiffany","yoona","yuri"]
    print name[pred]
    print
    for i,data in enumerate(name):
    print "class" + str(i+1) + " name：" + data
    print predictions[0][int(i)]*100
