import tensorflow as tf
import numpy as np
from utils import *
from imresize import imresize

class DCSRCN_InferbyCKPT(object):
    def __init__(self, sess, model_path, num_hiddens):
        self.sess = sess
        self.model_path = model_path
        self.num_hiddens = num_hiddens
        self.initialize()
        pass

    def initialize(self):
        new_saver = tf.train.import_meta_graph(self.model_path+".meta")
        new_saver.restore(self.sess,self.model_path)
        g = tf.get_default_graph()
        oper_list = g.get_operations()
        print(oper_list)

        self.image_test = g.get_tensor_by_name("images_test:0")
        self.preds_test = []
        for i in range(self.num_hiddens):
            self.preds_test.append(g.get_tensor_by_name("shared_model_1/block{}/add:0".format(i+1)))
        print(self.preds_test)

        #gd = g.as_graph_def()
        #tf.reset_default_graph()
        #g, = tf.import_graph_def(g, input_map={"ARCNN_FAST/image:0":self.image_test})

    def inference(self,input_img,scale,depth):
        if(np.max(input_img)>1): infer_image = (input_img/255).astype(np.float32)

        infer_image_scaled = imresize(input_img, scalar_scale=scale, output_shape=None)

        size = infer_image_scaled.shape
        if (len(infer_image_scaled.shape)==3): infer_image_input = infer_image_scaled[:,:,0].reshape(1, size[0], size[1], 1)
        else : infer_image_input = infer_image_scaled.reshape(1, size[0], size[1], 1)


        sr_img = self.sess.run(self.preds_test[depth], feed_dict={self.image_test: infer_image_input})
        #sr_img = np.expand_dims(sr_img,axis=-1)


        if (len(infer_image_scaled.shape) == 3) : infer_image_scaled[:,:,0] = sr_img[0,:,:,0]
        else : infer_image_scaled = sr_img[0]


        # output dim [w, d, c]
        return infer_image_scaled



if __name__ == "__main__":
    with tf.Session() as sess:
        model_path = "./checkpoint_291_branch2/demo/checks/checks.model-35"
        dcarcn = DCSRCN_InferbyCKPT(sess=sess,model_path=model_path,num_hiddens=6)
        img = np.random.normal(size=[100,100,3])
        print(dcarcn.inference(img,scale=1,depth=3))