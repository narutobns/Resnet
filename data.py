import tensorflow as tf
import os
import cv2
import tqdm
import pandas as pd
import numpy as np


def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def img_to_bytes(imgpath):
    with tf.gfile.FastGFile(imgpath,'rb') as f:
         encoded_jpg=f.read()
    return encoded_jpg 

'''
def data_split(labelfile):
    my_matrix=np.loadtxt(open(labelfile),delimiter=",",skiprows=0)
    x,y=my_matrix[:,:-1],my_matrix[:,-1]
    
    def iris_type(s):
        class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
        return class_label[s]
    data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    train=np.column_stack((x_train,y_train))
    np.savetxt('train.csv',train,delimiter=',')
    test=np.column_stack((x_test,y_test))
    np.savetxt('test.csv',test,delimiter=',')
'''

def data_split(labelfile):
    train=pd.DataFrame()
    val=pd.DataFrame()
    name_label=pd.read_csv(labelfile)
    cat_counts=name_label[' label'].value_counts()
    for cat in cat_counts.index:
        tmp=name_label[name_label[' label']==cat]
        train=train.append(tmp.iloc[0:int(0.8*cat_counts[cat])])
        val=val.append(tmp.iloc[int(0.8*cat_counts[cat]):])
    train.to_csv('train.csv',index=False)
    val.to_csv('val.csv',index=False)
    
def image_to_record(imagedir,labelfile,save_path):
    name_label=pd.read_csv(labelfile)
    namelist=name_label['name'].tolist()
    labellist=name_label[' label'].tolist()
    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer=tf.python_io.TFRecordWriter(path=save_path,options=writer_options)
    for name,label in tqdm.tqdm(zip(namelist,labellist)):
      example=tf.train.Example(features=tf.train.Features(feature={
       'img_name':_bytes_feature(name.encode()),
       'img_raw':_bytes_feature(img_to_bytes(os.path.join(imagedir,name))),
       'label':_float_feature(label)
      }))
      writer.write(example.SerializeToString())
    writer.close()    

def read_tfrecord(record_file,num_epochs=None,shuffle=True):
    filename_queue=tf.train.string_input_producer([record_file],num_epochs,shuffle)
    options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    reader=tf.TFRecordReader(options=options)
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,features={
        'img_name':tf.FixedLenFeature([],tf.string),
        "label":tf.FixedLenFeature([],tf.float32),
        "img_raw":tf.FixedLenFeature([],tf.string)
    })
    image=tf.image.decode_jpeg(features['img_raw'],channels=3)
    image=tf.image.resize_images(image,[224,448])
    img_mean=tf.constant([[[129.575]], [[127.99]], [[119.87]]],dtype=tf.float32)
    img_mean=tf.reshape(img_mean,[1,1,3])
    image=image-img_mean
    #image=tf.image.random_flip_left_right(image)
    #image=tf.image.random_brightness(image,32./255.)
    #image=tf.image.random_saturation(image,lower=0.4,upper=1.2)
    #image=tf.image.random_hue(image,0.2)
    #image=tf.image.randim_contrast(image,lower=0.4,upper=1.2)
    image_name=tf.cast(features['img_name'],tf.string)
    label=tf.cast(features['label'],tf.float32)
    return image_name,image,label

def read_tfrecord_without_label(record_file,num_epochs=None,shuffle=True):
    filename_queue=tf.train.string_input_producer([record_file],num_epochs,shuffle)
    options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    reader=tf.TFRecordReader(options=options)
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,features={
      'img_name':tf.FixedLenFeature([],tf.string),
      "img_raw":tf.FixedLenFeature([],tf.string)
    })
    image=tf.image.decode_jpeg(features['img_raw'],channels=3)
    image=tf.image.resize_images(image,[224,448])
    img_mean=tf.constant([[[129.575]], [[127.99]], [[119.87]]],dtype=tf.float32)
    img_mean=tf.reshape(img_mean,[1,1,3])
    image=image-img_mean
    image_name=tf.cast(features['img_name'],tf.string)
    return image_name,image

def img_to_record(img_dir,save_path):
    writer_options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer=tf.python_io.TFRecordWriter(path=save_path,options=writer_options)    
    for name in tqdm.tqdm(os.listdir(img_dir)):
      example=tf.train.Example(features=tf.train.Features(feature={
       'img_name':_bytes_feature(name.encode()),
       'img_raw':_bytes_feature(img_to_bytes(os.path.join(img_dir,name))),
      }))
      writer.write(example.SerializeToString())
    writer.close()  

def image_to_df(img_dir):
    image_df=pd.DataFrame(columns=['img_name','height','width'])
    for img in tqdm.tqdm(os.listdir(img_dir)):
        image=cv2.imread(os.path.join(img_dir,img),cv2.IMREAD_COLOR)
        height,width=image.shape[:2]
        image_df=image_df.append({'img_name':img,'height':height,'width':width},ignore_index=True)
    image_df.to_csv('im.csv',index=False)
