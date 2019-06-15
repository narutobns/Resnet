import os
import tqdm
from data import read_tfrecord
from resnet import resnet_backbone
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_blocks=[3,3,3,3]
classes=[0.1,0.2,0.5,1.,2.,5.,10.,50.,100.]
batch_size=32

def data_batch(recordfile,batch_size):
    _,img_data,label=read_tfrecord(recordfile)
    img_batch,label_batch=tf.train.shuffle_batch([img_data,label],batch_size=batch_size,num_threads=4,capacity=50000,min_after_dequeue=10000)
    return img_batch,label_batch

def loss_fun(logits,label):
    cross_entropy= tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=label)
    loss=tf.reduce_mean(cross_entropy)
    return loss

def accuracy(pred_class,label):
    accuracy=tf.reduce_mean(tf.cast(tf.equal(pred_class,label),dtype=tf.float32))
    return accuracy

def one_hot_label(label_batch,classes,batch_size):
    num_class=len(classes)
    classes=tf.constant(classes)
    bool_idx=tf.equal(classes,label_batch[0])
    idx=tf.where(bool_idx)
    index=tf.concat(idx,axis=0)
    i=1
    while i<batch_size:
      bool_idx=tf.equal(classes,label_batch[i])
      idx=tf.where(bool_idx)
      index=tf.concat([index,idx],axis=1)
      i=i+1
    index=tf.reshape(index,[batch_size])
    one_hot_label=tf.one_hot(index,num_class)
    return index,one_hot_label

if __name__=='__main__':
   num_class=len(classes)
   img_batch,label_batch=data_batch('./train.record',batch_size)
   logits,prob=resnet_backbone(img_batch,num_blocks,num_class,training=True)
   pred_class=tf.argmax(prob,axis=1)
   gt_class,gt_label=one_hot_label(label_batch,classes,batch_size)
   loss=loss_fun(logits,gt_label)
   acc=accuracy(pred_class,gt_class)

   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
       train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
   '''
   var_list = tf.trainable_variables()
   g_list = tf.global_variables()
   bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
   bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
   var_list += bn_moving_vars
   '''
   saver=tf.train.Saver(max_to_keep=5)
   
   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       
       coord = tf.train.Coordinator()
       threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
       for i in range(2881):
          _,loss_,acc_=sess.run([train_step,loss,acc])
          if i%40==0:
           #pred_class=[classes[indices_[i]] for i in range(batch_size)]
           #pred_class=tf.constant(pred_class)
           print('step:{},loss:{},accuracy:{}'.format(i,loss_,acc_))
           saver.save(sess,'./model/model',global_step=i,write_meta_graph=False)
       coord.request_stop()
       coord.join(threads) 
