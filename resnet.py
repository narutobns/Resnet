import tensorflow as tf
from tensorflow.contrib.layers import conv2d 
from tensorflow.contrib.layers import batch_norm as bn

def conv2d_bn(inputs,ch_out,kernel,training,scope,stride=1,padding='SAME',):
   with tf.variable_scope(scope): 
       x=conv2d(inputs,ch_out,kernel,stride,padding,activation_fn=None,scope='conv')
       x=bn(x,0.9,scale=True,scope='bn',is_training=training)
       x=tf.nn.relu(x)
   return x

def resnet_bottleneck(inputs,ch_out,stride,training):
    shortcut=inputs
    n_in=inputs.shape[3]
    if n_in!=ch_out*4:
       shortcut=conv2d_bn(shortcut,ch_out*4,[1,1],training,'shortcut',stride)
    conv_1=conv2d_bn(inputs,ch_out,[1,1],training,'conv1')
    if stride==2:
      conv_1=tf.pad(conv_1,[[0,0],[0,1],[0,1],[0,0]])
      conv_2=conv2d_bn(conv_1,ch_out,[3,3],training,'conv2',2,'VALID')
    else:
      conv_2=conv2d_bn(conv_1,ch_out,[3,3],training,'conv2')
    conv_3=conv2d(conv_2,ch_out*4,[1,1],activation_fn=None,scope='conv3')
    x=bn(conv_3,0.9,scale=True,is_training=training)
    x=tf.add(x,shortcut)
    x=tf.nn.relu(x)
    return x
    
def resnet_group(name,x,ch_out,count,stride,training):
    with tf.variable_scope(name):
         for i in range(0,count):
             with tf.variable_scope('block{}'.format(i)):
                 x=resnet_bottleneck(x,ch_out,stride if i==0 else 1,training)
    return x

def resnet_backbone(inputs,num_blocks,num_class,training):
    with tf.variable_scope('c1'):
        x=tf.pad(inputs,[[0,0],[2,3],[2,3],[0,0]])
        x=conv2d_bn(x,64,[7,7],training,stride=2,padding='VALID',scope='c1')
        x=tf.pad(x,[[0,0],[0,1],[0,1],[0,0]])
        x=tf.nn.max_pool(x,[1,3,3,1],[1,2,2,1],'VALID')
    c2=resnet_group('group0',x,64,num_blocks[0],1,training)
    c3=resnet_group('group1',c2,128,num_blocks[1],2,training)
    c4=resnet_group('group2',c3,256,num_blocks[2],2,training)
    c5=resnet_group('group3',c4,512,num_blocks[3],2,training)
    x=tf.reduce_mean(c5,[1,2]) #global avg_pool shape:[N,2048]
    logits=tf.layers.dense(x,num_class)
    prob=tf.nn.softmax(logits)
    return logits,prob
