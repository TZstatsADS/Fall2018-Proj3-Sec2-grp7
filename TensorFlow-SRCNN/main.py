import random
import tensorflow as tf
from model import SRCNN

class this_config():
    def __init__(self, is_train=True):
        self.epoch = 1
        self.image_size = 32
        self.label_size = 20
        self.c_dim = 3
        self.is_train = is_train
        self.scale = 3
        self.stride = 21
        self.checkpoint_dir = "checkpoint1"
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.result_dir = 'result'
        self.test_img = '' # Do not change this.
        
arg = this_config()
print("Hello TA!  We are group 7. Thank you for your work for us. Hope you have a happy day!")

with tf.Session() as sess:
    FLAGS = arg
    srcnn = SRCNN(sess,
                  image_size = FLAGS.image_size,
                  label_size = FLAGS.label_size,
                  c_dim = FLAGS.c_dim)
    srcnn.train(FLAGS)
    
    # Testing
    files = glob.glob(os.path.join(os.getcwd(), 'train_set', 'LR0', '*.jpg'))
    test_files = random.sample(files, len(files)//5)
    
    FLAGS.is_train = False
    count = 1
    for f in test_files:
        FLAGS.test_img = f
        print('Saving ', count, '/', len(test_files), ': ', FLAGS.test_img, '\n')
        count += 1
        srcnn.test(FLAGS)