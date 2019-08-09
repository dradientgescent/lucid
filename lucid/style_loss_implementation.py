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
  model_path = 'model_res.pb'
  image_shape = [None, 4, 128, 128, 128]
  image_value_range = (0, 1)
  input_name = 'input_1'


tumor_model = Tumor_Mode()
tumor_model.load_graphdef()

# graph_file = "model/segmentation_1.pb"
# graph_def = tf.GraphDef()

# with open(graph_file, "rb") as f:
#  graph_def.ParseFromString(f.read())

JITTER = 8
ROTATE = 10
SCALE = 1.8
L1 = -0.05
TV = -0.25
BLUR = -1.0

DECORRELATE = True



with tf.Graph().as_default() as graph, tf.Session() as sess:


  gram_template = tf.constant(np.load('/home/parth/lucid/lucid/GramTemplate.npy'),
                              dtype=tf.float32)
  print(gram_template.eval())
  obj = objectives.channel("conv2d_17/convolution", 3, gram=gram_template)
  # obj += L1 * objectives.L1(constant=.5)
  # obj += TV * objectives.total_variation()
  # obj += BLUR * objectives.blur_input_each_step()

  transforms = [
    transform.pad(2 * JITTER),
    transform.jitter(JITTER),
    # transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    transform.random_rotate(range(-ROTATE, ROTATE + 1))
  ]
  T = render.make_vis_T(tumor_model, obj,
                        param_f=lambda: param.image(240, channels=4, fft=DECORRELATE,
                                                    decorrelate=DECORRELATE),
                        optimizer=None,
                        transforms=transforms, relu_gradient_override=True, extras = gram_template)
  tf.initialize_all_variables().run()
  pprint([v.name for v in tf.get_default_graph().as_graph_def().node])
  for i in range(500):
    T("vis_op").run()

  print(T("input").name)
  print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0])
  print(T("input"))
  #show(np.hstack(T("input").eval()))
  # print(np.hstack(T("input").eval()))

  for i in range(1, 5):
    plt.subplot(1, 4, i)
    image = T("input").eval()[:, :, :, i - 1].reshape((240, 240))
    print(image.min(), image.max())
    plt.imshow(T("input").eval()[:, :, :, i - 1].reshape((240, 240)), cmap='gray',
               interpolation='bilinear', vmin=0., vmax=1.)

    # show(np.hstack(T("input").eval()))
plt.show()
