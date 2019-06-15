import os
import cv2
import tqdm
import numpy as np
import tensorflow  as tf

def cpu_mean(imgdir):
   count=0
   sum_r=0
   sum_g=0
   sum_b=0
   for filename in tqdm.tqdm(os.listdir(imgdir)):
      img=cv2.imread(os.path.join(imgdir,filename),cv2.IMREAD_COLOR)
      img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      img=cv2.resize(img,(224,448))
      sum_r=sum_r+img[:,:,0].mean()
      sum_g=sum_g+img[:,:,1].mean()
      sum_b=sum_b+img[:,:,2].mean()
      count=count+1
   sum_r=sum_r/count
   sum_g=sum_g/count
   sum_b=sum_b/count
   img_mean=[sum_r,sum_g,sum_b]
   return img_mean

def gpu_mean(imgdir):
   count=tf.constant(0.0)
   sum_r=tf.constant(0.0)
   sum_g=tf.constant(0.0)
   sum_b=tf.constant(0.0)
   for file in tqdm.tqdm(os.listdir(imgdir)):
      with tf.gfile.FastGFile(os.path.join(imgdir,file),'rb') as f:
           encode_jpg=f.read()
           img=tf.image.decode_jpeg(encode_jpg,channels=3)
           img=tf.image.resize_images(img,[224,224])
           sum_r=tf.add(sum_r,tf.reduce_mean(img[:,:,0]))
           sum_g=tf.add(sum_g,tf.reduce_mean(img[:,:,1]))
           sum_b=tf.add(sum_b,tf.reduce_mean(img[:,:,2]))
           count=count+1
   sum_r=tf.div(sum_r,count)
   sum_g=tf.div(sum_g,count)
   sum_b=tf.div(sum_b,count)
   return sum_r,sum_g,sum_b
      
if __name__=='__main__':
   img_mean=cpu_mean('./rmbimg/public_test_data')
   print(img_mean)
'''
   with tf.Session() as sess:
      r,g,b=gpu_mean('./rmbimg/train_data')
      print(r.eval(),g.eval(),b.eval())
'''
