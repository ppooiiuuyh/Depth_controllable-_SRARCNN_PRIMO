import os
import time
import tensorflow as tf
from utils import *
from imresize import *
from metrics import *
import matplotlib.pyplot as plt
import pprint
import math
import numpy as np
import sys
import glob
from tqdm import tqdm


class DCARCNN(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.preprocess()
        self.model()
        self.other_tensors()
        self.init_model()


# ==========================================================
# preprocessing
# ==========================================================
    def preprocess(self):
        self.train_label = []
        self.train_input = []
        self.test_label = []
        self.test_input = []
        if self.args.type == "YCbCr" : input_setup = input_setup_demo
        elif self.args.type == "RGB" : input_setup = input_setup_demo
        else : input_setup = input_setup_demo

        # scale augmentation
        scale_temp = self.args.scale
        for s in [1, 0.9, 0.8, 0.7, 0.6]:
            self.args.scale = s
            train_input_, train_label_ = input_setup(self.args, mode="train")
            self.train_label.extend(train_label_)
            self.train_input.extend(train_input_)
        self.args.scale = scale_temp

        # augmentation (rotation, miror flip)
        self.train_label = augumentation(self.train_label)
        self.train_input = augumentation(self.train_input)

        # setup test data
        self.test_input, self.test_label = input_setup(self.args, mode="test")
        pass



# ==========================================================
# build model
# ==========================================================
    def model(self):
        with tf.variable_scope("ARCNN_FAST") as scope:
            shared_inner_model_template = tf.make_template('shared_model', self.inner_model)
            #self.images = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='images')

            self.images = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.args.patch_size, self.args.patch_size, self.args.c_dim],  name='labels')
            self.pred = shared_inner_model_template(self.images)

            #self.image_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='images_test')
            self.image_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='image_test')
            self.label_test = tf.placeholder(tf.float32, [1, None, None, self.args.c_dim], name='labels_test')
            self.pred_test = shared_inner_model_template(self.image_test)





# ===========================================================
# inner model
# ===========================================================
    def inner_model(self, inputs):
        # ----------------------------------------------------------------------------------
        # input layer
        # ------------------------------------------------------------------------------------------
        with tf.variable_scope("feature_extraction") as scope:
            conv_w = tf.get_variable("conv_w", [3, 3, self.args.c_dim, 64],  initializer=tf.contrib.layers.xavier_initializer())
            conv_b = tf.get_variable("conv_b", [64], initializer=tf.constant_initializer(0))
            layer = tf.nn.bias_add(tf.nn.conv2d(inputs, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
            alpha = tf.get_variable("prelu_alpha",[1],initializer=tf.constant_initializer(0.2))
            layer = tf.nn.leaky_relu(layer,alpha=alpha)

        with tf.variable_scope("inner_layer") as scope:
            for i in range(self.args.num_hidden):
                conv_w = tf.get_variable("conv_w"+str(i), [3, 3, 64, 64],  initializer=tf.contrib.layers.xavier_initializer())
                conv_b = tf.get_variable("conv_b"+str(i), [64], initializer=tf.constant_initializer(0))
                layer = tf.nn.bias_add(tf.nn.conv2d(layer, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
                alpha = tf.get_variable("prelu_alpha"+str(i),[1],initializer=tf.constant_initializer(0.2))
                layer = tf.nn.leaky_relu(layer,alpha=alpha)

        with tf.variable_scope("reconstruction") as scope:
            conv_w = tf.get_variable("conv_w", [3, 3,64, self.args.c_dim],   initializer=tf.contrib.layers.xavier_initializer())
            conv_b = tf.get_variable("conv_b", [self.args.c_dim], initializer=tf.constant_initializer(0))
            layer = tf.nn.bias_add(tf.nn.conv2d(layer, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)


        pred =  layer +inputs
        return pred
    # ----------------------------------------------------------------------------------------





# ============================================================
# other tensors related with training
# ============================================================
    def other_tensors(self):
        with tf.variable_scope("trainer") as scope:
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.loss = tf.reduce_mean(tf.square(self.pred - self.labels))  # L1 is betther than L2
            self.learning_rate = tf.maximum(tf.train.exponential_decay(self.args.base_lr, self.global_step,
                                                                           len(self.train_label) // self.args.batch_size * self.args.lr_step_size,
                                                                           self.args.lr_decay_rate,
                                                                           staircase=True),
                                                self.args.min_lr)  # stair case showed better result

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            # tensor board
            self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)
            self.loss_history = tf.summary.scalar("loss", self.loss)
            self.summary = tf.summary.merge_all()
            self.psnr_history = []
            self.ssim_history = []



# ============================================================
# init tensors
# ============================================================
    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)
        if self.cpkt_load(self.args.checkpoint_dir, self.args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def cpkt_save(self, checkpoint_dir, step):
        model_name = "checks.model"
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def cpkt_load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if checkpoint_itr == 0:
            print("train from scratch")
            return True

        elif checkpoint_dir == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)

        else:
            ckpt = os.path.join(checkpoint_dir, "checks.model-" + str(checkpoint_itr))

        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False





# ==========================================================
# functions
# ==========================================================
    def inference(self, input_img):
        if (np.max(input_img) > 1): input_img = (input_img / 255).astype(np.float32)

        size = input_img.shape
        if (len(input_img.shape) == 3):
            infer_image_input = input_img[:, :, 0].reshape(1, size[0], size[1], 1)
        else:
            infer_image_input = input_img.reshape(1, size[0], size[1], 1)
        sr_img = self.sess.run(self.pred_test, feed_dict={self.image_test: infer_image_input})
        # sr_img = np.expand_dims(sr_img,axis=-1)


        #input_img = imresize(input_img,self.args.scale)
        if (len(input_img.shape) == 3):
            input_img[:, :, 0] = sr_img[0, :, :, 0]
        else:
            input_img = sr_img[0]

        return input_img #return as ycbcr




# ==========================================================
# train
# ==========================================================
    def train(self):
        self.test()
        print("Training...")
        start_time = time.time()


        for ep in range(self.args.epoch):
            # =============== shuffle and prepare batch images ============================
            seed = int(time.time())
            np.random.seed(seed); np.random.shuffle(self.train_label)
            np.random.seed(seed); np.random.shuffle(self.train_input)

            #================ train rec ===================================================
            batch_idxs = len(self.train_label) // self.args.batch_size
            for idx in tqdm(range(0, batch_idxs)):
                batch_labels = np.expand_dims(np.array(self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)
                batch_inputs = np.expand_dims(np.array(self.train_input[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)

                feed = {self.images: batch_inputs, self.labels:batch_labels}
                _, err, lr, summary = self.sess.run( [self.train_op, self.loss, self.learning_rate, self.summary], feed_dict=feed)
                self.summary_writer.add_summary(summary,self.global_step.eval())



            #=============== print log =====================================================
            if ep % 1 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_com: [%.8f], lr: [%.8f]" \
                      % ((ep + 1), self.global_step.eval(), time.time() - start_time, np.mean(err), lr))
                self.test()


            #================ save checkpoints ===============================================
            if ep % self.args.save_period == 0:
                self.cpkt_save(self.args.checkpoint_dir, ep + 1)


# ==========================================================
# test
# ==========================================================
    def test(self):
        print("Testing...")
        psnrs_preds = []
        ssims_preds = []

        preds = []
        labels = []
        images = []

        for idx in range(0, len(self.test_label)):
            test_label = np.array(self.test_label[idx]) #none,none,3
            test_input = np.array(self.test_input[idx])

            # === original =====
            for f in [5, 10, 20, 40, 50, 60, 80, 90, 100]:
                cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori" + str(f) + ".jpg"),
                            (ycbcr2rgb(test_label)*255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), f])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori.PNG"), (ycbcr2rgb(test_label)*255)[..., ::-1])
            # ==================
            result = self.inference(test_input)
            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "rec"+str(self.args.jpgqfactor)+".bmp"), (ycbcr2rgb(result)*255)[...,::-1])

            preds.append(result)
            labels.append(test_label)


        # cal PSNRs for each images upscaled from different depths
        for i in range(len(self.test_label)):
            if len(np.array(labels[i]).shape)==3 : labels[i] = np.array(labels[i])[:,:,0]
            if len(np.array(preds[i]).shape)==3 : preds[i] = np.array(preds[i])[:,:,0]
            psnrs_preds.append(psnr(labels[i], preds[i], max=1.0, scale=self.args.scale))
            ssims_preds.append(ssim(labels[i], preds[i], max=1.0, scale=self.args.scale))

        # print evalutaion results
        print("===================================================================================")
        print("PSNR: " + str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)) + "dB")
        print("SSIM: " + str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print("===================================================================================")

        self.psnr_history.append(str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)))
        self.ssim_history.append(str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print()
        print(self.psnr_history)
        print(self.ssim_history)
