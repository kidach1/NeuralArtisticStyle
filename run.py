from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iterations = 30

# 各種パスの準備
base_image_path = 'input.jpg'
style_reference_image_path = 'output.jpg'
result_prefix = ''

# それぞれの損失の重み
total_variation_weight = 1.
style_weight = 1.
content_weight = 0.025

# 生成画像のサイズ
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)


# 画像前処理。リサイズと適切なフォーマット（テンソル）に変更
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# 画像後処理。テンソルから画像へ
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # ピクセルの平均を0にする
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 前処理によって画像のテンソル表現を取得
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# channels_firstならチャネル(3)を先に、channels_lastなら後に。
# backendがtheanoならchannels_first, tfならchannels_last
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# base_image（コンテンツ画像）、style_reference_image（スタイル画像）、combination_image（生成画像）を一つのテンソルに
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)


# input_tensorを入力としてVGG19ネットワークを構築（ImageNetで訓練済み）
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model loaded.')
#print(model.summary())

# 各層（block{n}_conv{n}）ごとの名前と出力のdictionary
#print(model.layers[1].name)
#print(model.layers[1].output)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


# スタイル損失の計算に必要な関数

# グラム行列 (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    #print(features)
    gram = K.dot(features, K.transpose(features))
    return gram


# スタイル損失関数
# 特徴マップ（各層に対してフィルタを通して生成された結果）ごとのグラム行列（つまり、画像中に含まれる特徴間の相関）、
# すなわち「画風を表すベクトル」について、スタイル画像とcombination画像（生成画像）それぞれ計算し、
# 2つの損失を取って最小化する
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
#    return K.sum(K.square(S - C)) / (2. * (channels ** 2) * (size ** 2))


# コンテンツ損失関数
# 生成画像とコンテンツ画像間の損失を取る
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

# 画像を滑らかにする制約としての3つめの損失関数
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
  
  
# コンテンツ損失の計算（コンテンツに関しては出力に近い抽象的な層で損失を取る）
# combine these loss functions into a single scalar
loss = K.variable(0.)
#layer_features = outputs_dict['block5_conv1']
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features, combination_features)


# スタイル損失の計算（スタイルに関しては各畳み込み層すべてで損失を取る）
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    # outputs_dictは層ごとのbase_image,style_reference_image,combination_imageのテンソルなので
    # 1でstyle_reference_imageのfeatureを取得
    style_reference_features = layer_features[1, :, :, :]
    # 同上より2でcombination_imageのfeatureを取得
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# 画像を滑らかにする3つめの損失関数    
loss += total_variation_weight * total_variation_loss(combination_image)


# combintion_imageのlossに関する勾配を求める
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

# Keras functionを用いて、combination_imageが与えられた時、それに応じたoutputsを返す関数を定義（内部実装が気になる）
f_outputs = K.function([combination_image], outputs)

from keras.models import Model

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))

    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
        
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# スタート時点での生成画像。本来ランダム画像を用いる
#x = np.random.random((1, 400, 400, 3)) * 20 + 128.
x = preprocess_image(base_image_path)


from scipy.misc import imsave

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    
    plt.figure(figsize=(2.0,2.0))
    plt.grid(False)
    fig = plt.imshow(deprocess_image(x.copy()))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    
    print('Current loss value:', min_val)
    img = deprocess_image(x.copy())
    fname = result_prefix + 'output_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

