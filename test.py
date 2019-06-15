import os
import tqdm
import pandas as pd
import tensorflow as tf
from data   import read_tfrecord
from train  import accuracy,num_blocks,classes,batch_size,one_hot_label
from resnet import resnet_backbone

os.environ['CUDA_VISIBLE_DEVICES']='1'

if __name__=='__main__':
   num_class=len(classes)
   pred_result=pd.DataFrame(columns=['name','label'])
   img_name,img_data=read_tfrecord_without_label('./test.record',num_epochs=1,shuffle=False)
   img_name,img_batch=tf.train.batch([img_name,img_data],1)
   logits,prob=resnet_backbone(img_batch,num_blocks,num_class,training=False)
   pred_class=tf.argmax(prob,axis=1)
   
   saver=tf.train.Saver()
   with tf.Session() as sess:
       sess.run(tf.local_variables_initializer())
       coord=tf.train.Coordinator()
       threads=tf.train.start_queue_runners(sess=sess,coord=coord)
       ckpt=tf.train.get_checkpoint_state('./model/model/')
       if ckpt and ckpt.model_checkpoint_path:
         saver.restore(sess,'./model/model/resnet-2880')#ckpt.model_checkpoint_path)
         for n in tqdm.tqdm(range(20000)):
            i,img_name_=sess.run([pred_class,img_name])
            value=classes[i[0]]
            img_name_=img_name_[0].decode()
            pred_result=pred_result.append({'name':img_name_,'label':value},ignore_index=True)
       pred_result.to_csv('sub.csv',index=False)
       coord.request_stop()
       coord.join(threads)
