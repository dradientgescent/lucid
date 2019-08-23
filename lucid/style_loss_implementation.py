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
import matplotlib.gridspec as gridspec



def show_images(images):
  html = ""
  for image in images:
    data_url = _image_url(image)
    html += '<img width=\"100\" style=\"margin: 10px\" src=\"' + data_url + '\">'
  _display_html(html)

class Tumor_Mode(Model):
  model_path = 'U-Resnet.pb'
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
ROTATE = 4
SCALE = 1.3
L1 = -0.05
TV = -5e-7
BLUR = -1e-4

DECORRELATE = True

layer = 7
block = 1
channel = lambda n: objectives.channel("conv2d_9/convolution" , n, gram=gram_template)

fig = plt.figure()
plt.tight_layout()
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

layers_to_consider = [20]
indices_to_consider = [18]

for l, index in zip(layers_to_consider, indices_to_consider):

  channel = lambda n: objectives.channel("conv2d_%d/convolution" %l, n, gram=gram_template)

  with tf.Graph().as_default() as graph, tf.Session() as sess:


    gram_template = tf.constant(np.load('/home/parth/lucid/lucid/GramTemplate.npy'),
                                  dtype=tf.float32)
    print(gram_template.get_shape())

    obj = channel(index)
    # obj += L1 * objectives.L1(constant=.5)
    obj += TV * objectives.total_variation()
    #obj += BLUR * objectives.blur_input_each_step()

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
    # # print(np.hstack(T("input").eval()))
    # for i in range(1, 5):
    #     plt.subplot(1, 4, i)
    #     plt.imshow(T("input").eval()[:,:,:,i-1].reshape((240, 240)), cmap='gray', interpolation = 'bilinear', vmin = 0., vmax = 1.)

    plt.figure(figsize=(10, 40))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.025, hspace=0.05)
        
        
    for i in range(1, 5):
      ax = plt.subplot(gs[0, i-1])
      im = ax.imshow(T("input").eval()[:,:,:,i-1].reshape((240, 240)), cmap='gray', interpolation = 'bilinear', vmin = 0., vmax = 1.)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_aspect('equal')
      ax.tick_params(bottom='off', top='off', labelbottom='off' )
                # plt.subplot(7, 7, i*7 +(j+1))

    plt.savefig('consider/no_reg_%d_%d.png' %(l, index), bbox_inches='tight')
    # ax = fig.add_subplot(4, 4, l+1)
    # ax.set_title(('%d, %d' %(layer, l)))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # image = T("input").eval()[:, :, :, 0].reshape((128, 128))
    # print(image.min(), image.max())
    # plt.imshow(T("input").eval()[:, :, :, 0].reshape((128, 128)), cmap='gray',
    #            interpolation='bilinear', vmin=0., vmax=1.)

      # show(np.hstack(T("input").eval()))
  # plt.show()
