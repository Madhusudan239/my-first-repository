import cv2
import numpy as np
from deskew import determine_skew
from pytesseract import Output
import pytesseract
import imutils
from PIL import Image
import math
import boto3
import os
import json 
from datetime import datetime
import pandas as pd
from urllib.parse import unquote_plus

class PreprocessValidation():
    
    def __init__(self):
        
        self.s3r = boto3.resource("s3")

        self.s3c = boto3.client('s3')
        

        
    def preprocess_json_to_dataframe(self,preprocess_bucket_name,preprocess_folder_name):
        
        self.preprocess_bucket_name = preprocess_bucket_name
        # self.preprocess_bucket = self.s3r.Bucket(self.preprocess_bucket_name)
        # self.preprocess_bucket_image_obj = self.preprocess_bucket .objects.filter(Prefix = f'{preprocess_folder_name}/')
        self.preprocess_bucket_images = [preprocess_folder_name]
        # print(self.preprocess_bucket_images)
        for i in self.preprocess_bucket_images:
            obj = self.s3r.Object(self.preprocess_bucket_name, i)
            data = obj.get()['Body'].read().decode('utf-8')
            json_data = json.loads(data)
            df1 = pd.DataFrame(json_data)
            # test.append(df)
        # df1 = pd.concat(test)
        return df1
    
    def human_review_json_to_dataframe(self, human_review_bucket_name,human_review_folder_name):


        # df1 = pd.read_excel(r"D:\Zigna AI Corp\Zigna AI - Inprocess Data\PDF-processing\test images_human review\validation_test_1216772022.xlsx")
        self.human_review_bucket_name = human_review_bucket_name
        self.human_review_bucket = self.s3r.Bucket(self.human_review_bucket_name)
        self.human_review_bucket_image_obj = self.human_review_bucket.objects.filter(Prefix = f'{human_review_folder_name}/')
        self.human_review_bucket_images = [i.key for i in self.human_review_bucket_image_obj]
        print(self.human_review_bucket_images)

        for i in self.human_review_bucket_images:
            df1 = pd.DataFrame()
            obj = self.s3r.Object(self.human_review_bucket_name, i)
            data = obj.get()['Body'].read().decode('utf-8')
            json_data = json.loads(json.dumps(data))
            json_data_split = json_data.split('\n')
            details  = []
        
            for j in json_data_split:
                if len(j) > 0:
                    d = json.loads(j)
                    key = list(d.keys())
                    details.append((d.get(key[0]),d.get(key[1])))
            df = pd.DataFrame(details, columns = ['src_path','metadata'])
            df1 = pd.concat([df1, df])
        return df1
    
    def change_skew(self, image, angle, background):
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
        
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(255,255,255))
   
    def skew_correction(self, rgb): 
        
        angle = determine_skew(rgb)
        if angle == 0.0:
            self.skew = rgb 
            self.report_json['metadata'].append(10)
        elif 14 in self.report_json['metadata'] or 15 in self.report_json['metadata']:
            self.skew = rgb
            return self.skew, self.report_json
        else:
            self.skew = self.change_skew(rgb, angle, (0, 0, 0))
            if str(angle) > str(0):
                self.report_json['metadata'].append(8)
            elif str(angle) < str(0):
                self.report_json['metadata'].append(9)
        return self.skew, self.report_json
    
    def rotation_validation(self, img, x):
        self.rgb = imutils.rotate_bound(img,  angle= x)
        return self.rgb
    def skew_validation(self, rgb):
        self.rgb = self.change_skew(self.rgb, determine_skew(self.rgb), (0, 0, 0))
        return self.rgb
    
    def re_preprocess(self,path, metadata):

        imgpath = path.replace('s3://human-review-stage/revmaxai-images/', 'human-review/')
        self.img = self.preprocess_bucket.Object(imgpath).get().get('Body').read()
        
        self.rgb = cv2.imdecode(np.asarray(bytearray(self.img)), 0)
        metadata = eval(metadata)
        print(type(metadata))
        for md in metadata:
            print(md)
            self.d = {0:90,1:270,2:0,3:180,4:90,5:270,6:0,7:180}
            if md in [0,1,2,3,4,5,6,7]:
                self.rgb = self.rotation_validation(self.rgb, self.d.get(md))
            elif md in [8,9] :
                self.rgb = self.skew_validation(self.rgb)
            else:
                self.rgb = self.rgb

        data_serial = cv2.imencode('.png', self.rgb)[1].tobytes()
        self.s3c.put_object(Body=data_serial,Bucket= self.preprocess_bucket_name,Key=imgpath.replace('human-review','preprocessed'), ContentType='png')
        print('file recorrrected and saved in processed')
        self.s3c.delete_object(Bucket = self.preprocess_bucket_name,  Key = imgpath)

    def validate(self,preprocess_bucket_name,preprocess_folder_name, human_review_bucket_name,human_review_folder_name):
        
        a = self.preprocess_json_to_dataframe(preprocess_bucket_name, preprocess_folder_name)
        a = a.reset_index(drop = True)

        b = self.human_review_json_to_dataframe(human_review_bucket_name, human_review_folder_name)
        c = a.merge(b, on=['src_path'], how='right')
        c = c[['metadata_y', 'src_path']]
        c.rename(columns = {'metadata_y':'metadata'}, inplace = True)
        c = c.reset_index(drop = True)
        for n1, i in enumerate(a['src_path']):
            for n2, j in enumerate(c['src_path']):
                    
                if str(i) == str(j):
                    if a.loc[n1,'metadata'] == c.loc[n2,'metadata']:
                        self.s3c.copy_object(CopySource={'Bucket': self.preprocess_bucket_name, 'Key': j.replace('s3://human-review-stage/revmaxai-images', 'intermediate')}, Bucket = self.preprocess_bucket_name, Key= j.replace('s3://human-review-stage/revmaxai-images','processed'))
                        self.s3c.delete_object(Bucket = self.preprocess_bucket_name,  Key = j.replace('s3://human-review-stage/revmaxai-images', 'intermediate'))
                    else:
                        print(j)
                        self.re_preprocess(c['src_path'][n2], c['metadata'][n2])
                            
        return c, a
    
o = PreprocessValidation()


c,d= o.validate('human-review-stage', 'output-json','human-review-stage','flagged-images/output/preprocess-verification-flow-1-clone/manifests/output')

def lambda_handler(event, context):
    if event:
        file_obj = event["Records"][0]
        bucket_name = str(file_obj["s3"]["bucket"]["name"])
        file_name = unquote_plus(str(file_obj["s3"]["object"]["key"]))
        print(f"Bucket: {bucket_name} ::: Key: {file_name}")
        preprocess_folder_name = 'output-json'
        human_review_folder_name = file_name
        o= PreprocessValidation()
        c,d = o.validate(bucket_name, preprocess_folder_name, bucket_name, human_review_folder_name)
        print('madhu')
        print('unittest12345')

