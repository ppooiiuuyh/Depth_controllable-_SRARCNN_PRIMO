import tensorflow as tf
import numpy as np

class DCARCN_InferbyCKPT(object):
    def __init__(self, sess, model_path):
        self.sess = sess
        self.model_path = model_path
        self.initialize()
        pass

    def initialize(self):
        new_saver = tf.train.import_meta_graph(self.model_path+".meta")
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ARCNN_FAST")
        new_saver = tf.train.Saver(var_list=vars, max_to_keep=0)
        new_saver.restore(self.sess,self.model_path)
        g = tf.get_default_graph()
        oper_list = g.get_operations()
        #print(oper_list)

        self.image_test = g.get_tensor_by_name("ARCNN_FAST/image_test:0")
        self.pred_test = g.get_tensor_by_name("ARCNN_FAST/shared_model_1/add:0")
        print(self.pred_test)

        #gd = g.as_graph_def()
        #tf.reset_default_graph()
        #g, = tf.import_graph_def(g, input_map={"ARCNN_FAST/image:0":self.image_test})

    def inference(self,input_img):
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


if __name__ == "__main__":
    with tf.Session() as sess:
        model_path = "./checkpoint_l4_jf20/YCbCr/checks/checks.model-80"
        dcarcn = DCARCN_InferbyCKPT(sess=sess,model_path=model_path)
        img = np.random.normal(size=[100,100,3])
        print(dcarcn.inference(img))