#Caffeを使って少女時代の顔を識別するためのプログラム
* Cifar10で少女時代のメンバーを学習。  
* このプログラムは`caffe/examples/cifar10/`上で動作させることを想定
* 学習器の設定はこのレポジトリにある`snsd_cifar10_full.prototxt`、Caffeモデルは`snsd_cifar10_full_150717_iter_60000.caffemodel`を指定
* 平均画像ファイルは`snsd_mean.binaryproto`

##snsd_classify.py
* 顔を切り取った写真を前提
* 一番もっともらしいメンバーの名前を表示
* その他のメンバーの確率も表示

###実行方法
```bash
cd python
python snsd_classify.py src.jpg
```

##snsd_facedetection.py
* OpenCVで入力画像から顔検出
* そのあと切り出した画像から、分類器にかける
* 1番確率の高かったメンバー名を画像に書きこむ
* 2番目，3番目の候補も下に表示

###実行方法
```bash
cd python
python snsd_facedetection.py src.jpg output.jpg
```
または
```bash
cd python
python snsd_facedetection.py src.jpg
```
出力先を指定しない場合は保存した時間を名前にして勝手に`success_img`というディレクトリにぶち込まれます．

実行すると「Show Image」と書かれたウィンドウが出現し，結果が可視化される．  
この結果を保存したかったらウィンドウ上で`s`キーを押す．保存したくなかったら`s`以外のキーを押すとプログラムが終了．

###実行結果
この結果は**データの水増し，Dropoutあり（後述）**の手法で学習させた結果です．

![yoona_sooyoung.jpg](success_img/yoona_sooyoung.jpg)

##Dropoutをいれた
Cifar10のモデルをそのまま使ったら，過学習が起きたので，これを軽減させるために，データをランダムにクロップして学習につかったり，画像を左右反転させたり，データの水増しをした．
あとDropoutと呼ばれるブロックを追加した．
このモデルは[dropout](https://github.com/Ry0/snsd_classify/tree/dropout)ブランチにそれ用の`.prototxt`があります．

###ノーマル
* バリバリ過学習が起きてる．
* テストデータに関する精度も77%程度

![overtraining.png](plot/overtraining/overtraining.png)

###データの水増し，Dropoutあり
* 過学習が抑えられている！！
* テストデータに関する精度も82%とノーマルのときよりも精度向上＼(^o^)／

![dropout.png](plot/dropout/dropout.png)