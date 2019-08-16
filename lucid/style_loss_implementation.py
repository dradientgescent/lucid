import optvis.param as param
import optvis.render as render
from misc.io.showing import _image_url, _display_html
from modelzoo.vision_base import Model
import tensorflow as tf
import optvis.transform as transform
from optvis import objectives
from misc.io import show
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pprint import pprint

def show_images(images):
  html = ""
  for image in images:
    data_url = _image_url(image)
    html += '<img width=\"100\" style=\"margin: 10px\" src=\"' + data_url + '\">'
  _display_html(html)

class Tumor_Mode(Model):
  model_path = 'model_dense.pb'
  image_shape = [None, 4, 128, 128, 128]
  image_value_range = (0, 1)
  input_name = 'input_1'


tumor_model = Tumor_Mode()
tumor_model.load_graphdef()

# graph_file = "model/segmentation_1.pb"
# graph_def = tf.GraphDef()

# with open(graph_file, "rb") as f:
#  graph_def.ParseFromString(f.read())

JITTER = 4
ROTATE = 8
SCALE = 1.3
L1 = -0.05
TV = -0.25
BLUR = -1.0

DECORRELATE = True

gram_template = tf.constant(np.load('/home/parth/lucid/lucid/test_image.npy'),
                              dtype=tf.float32)
print(gram_template.get_shape())
layer = 7
block = 1
channel = lambda n: objectives.channel("conv2d_10/convolution" , n, gram=gram_template)

fig = plt.figure()
plt.tight_layout()
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

for l in range(30, 31):
  with tf.Graph().as_default() as graph, tf.Session() as sess:

    obj = channel(l)
    # obj += L1 * objectives.L1(constant=.5)
    # obj += TV * objectives.total_variation()
    # obj += BLUR * objectives.blur_input_each_step()

    transforms = [
      transform.pad(2 * JITTER),
      transform.jitter(JITTER),
      # transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
      transform.random_rotate(range(-ROTATE, ROTATE + 1))
    ]

    # Currently only works with
    T = render.make_vis_T(tumor_model, obj,
                          param_f=lambda: param.image(240, channels=4, fft=DECORRELATE,
                                                      decorrelate=DECORRELATE),
                          optimizer=None,
                          transforms=transforms, relu_gradient_override=True, extras = gram_template)
    tf.initialize_all_variables().run()

    for i in range(500):
      T("vis_op").run()

    #print(T("input").name)
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0].shape)
    #print(T("input"))
    #show(np.hstack(T("input").eval()))
    # print(np.hstack(T("input").eval()))
    for i in range(1, 5):
        plt.subplot(1, 4, i)
        plt.imshow(T("input").eval()[:,:,:,i-1].reshape((240, 240)), cmap='gray', interpolation = 'bilinear', vmin = 0., vmax = 1.)

    # ax = fig.add_subplot(4, 4, l+1)
    # ax.set_title(('%d, %d' %(layer, l)))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # image = T("input").eval()[:, :, :, 0].reshape((128, 128))
    # print(image.min(), image.max())
    # plt.imshow(T("input").eval()[:, :, :, 0].reshape((128, 128)), cmap='gray',
    #            interpolation='bilinear', vmin=0., vmax=1.)

      # show(np.hstack(T("input").eval()))
plt.show()
