#!/usr/bin/env python
# coding: utf-8

# pip install paddlehub     #--upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install shapely       #          -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install pyclipper     #          -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install paddlepaddle

# In[35]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import paddlehub as hub
import cv2


# In[36]:


save_path='out'
 
with open('test.txt', 'r') as f:
    test_img_path=[]
    for line in f:
        test_img_path.append(line.strip())
        
print("test_img_path:")
print(test_img_path)
print("\n")

#images
for file_name in test_img_path:
    print(file_name)
    img1 = mpimg.imread(file_name) 
    plt.figure(figsize=(5,5))
    plt.imshow(img1) 
    plt.axis('off') 
    plt.show()


# load model

# In[37]:


#path_list = [args['dataset'] + '/' + path for path in os.listdir(args['dataset'])]
#          
ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

np_images = [cv2.imread (image_path) for image_path in test_img_path] #         Test.txt Photo Path
 


# xử lý+save output_image

# In[38]:


results = ocr.recognize_text(
    images = np_images, # picture data, ndarray.shape is [H, W, C], BGR format;
    use_gpu = False, #Use GPU; if you use GPU, set the CUDA_VISIBLE_DEVICES environment variable first.
    output_dir = 'out', #the saving path of the # picture, is set to Ocr_Result
    visualization = True, #   Save the identification result as a picture file;
    box_thresh = 0.5, # detect the threshold of the text frame confidence;
    text_thresh = 0.5) # identifies the threshold of Chinese text confidence;
# or
# result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])


# show output

# In[39]:


#path = 'C:\\Users\\dovt5\\Downloads\\A-CRNN-model-for-Text-Recognition-in-Keras-master\\plz_work\\out'
path = os.path.abspath(os.getcwd())+"\\"+ save_path
print("path")
print(path)
out_img_path=[]
for root, directories, files in os.walk(path, topdown=False):
    for name in files:
        file_link=os.path.join(root, name)
        print(file_link)
        out_img_path.append(file_link)
        print("nen vo cai nay")
    for name in directories:
        file_link=os.path.join(root, name)
        print(file_link)
        out_img_path.append(file_link)
        print("ko nen vo cai nay")


for x in range(0, len(results)):
    file_name=out_img_path[x]
    print(file_name)
    img1 = mpimg.imread(file_name) 
    plt.figure(figsize=(100,100))
    plt.imshow(img1) 
    plt.axis('off') 
    plt.show()
    result=results[x]
    data = result['data']
    save_path = result['save_path']
    for infomation in data:
        print('text: ',
              infomation['text'],
              '\nconfidence: ', 
              infomation['confidence'],
              '\ntext_box_position: ', 
              infomation['text_box_position'])
    


# In[ ]:




