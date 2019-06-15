import os
import tqdm
import pandas as pd
import tensorflow as tf
from data   import read_tfrecord
from train  import accuracy,num_blocks,classes,batch_size,one_hot_label
from resnet import resnet_backbone

os.environ['CUDA_VISIBLE_DEVICES']='1'

if __name__=='__main__':
   err=0
   num_class=len(classes)
   img_name,img_data,label=read_tfrecord('./test.record',num_epochs=1,shuffle=False)
   img_name,img_batch,label_batch=tf.train.batch([img_name,img_data,label],1)
   logits,prob=resnet_backbone(img_batch,num_blocks,num_class,training=False)
   pred_class=tf.argmax(prob,axis=1)
   gt_class,_=one_hot_label(label_batch,classes,1)
   acc=accuracy(pred_class,gt_class)
   
   saver=tf.train.Saver()
   with tf.Session() as sess:
       sess.run(tf.local_variables_initializer())
       coord=tf.train.Coordinator()
       threads=tf.train.start_queue_runners(sess=sess,coord=coord)
       ckpt=tf.train.get_checkpoint_state('./model/model/')
       if ckpt and ckpt.model_checkpoint_path:
       #if ckpt and ckpt.all_model_checkpoint_paths:
       #for path in ckpt.all_model_checkpoint_paths:
       #saver.restore(sess,path)
         saver.restore(sess,'./model/model/resnet-2880')#ckpt.model_checkpoint_path)
        #try:
          #while not coord.should_stop():
            #global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
         for n in tqdm.tqdm(range(20000)):
            i,img_name_,acc_=sess.run([pred_class,img_name,acc])
            value=classes[i[0]]
            img_name_=img_name_[0].decode()
            if acc_==0.:
              err=err+1
              #print('Wrong image:{},Pred_class:{}'.format(img_name_,value))
        #finally:
       print('Accuracy:{},Wrong total:{}'.format(1-err/7928,err))
       coord.request_stop()
       coord.join(threads)
