# https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#22-create-labels
# https://yolov8.org/how-to-train-yolov8-on-custom-dataset/

from bing_image_downloader import downloader
from ultralytics import YOLO
import cv2
import os
import shutil


class YoloDataset():

    def __init__(self, yolo_weights="yolov8n-seg.pt", num_imgs=10, 
                 categories=["wojsko polskie", "ludzie"], output_dir="dataset", 
                 custom_dir="custom_dataset", train_val_split_ratio=0.6) -> None:
        self.yolo_model = YOLO(yolo_weights)
        self.num_imgs = num_imgs # number of images per category
        self.categories = categories # categories of images
        self.output_dir = output_dir # parent directory for images
        self.custom_dir = custom_dir # directory for yolo dataset
        self.train_val_split_ratio = train_val_split_ratio
        self.original_cwd = os.getcwd()
        self.global_index = len(self.categories)*num_imgs*10 # initial value of global index
    
    def scrap_images(self, query, limit, output_dir):
        # method scraps the web for images matching query
        downloader.download(query=query, limit=limit, 
                            output_dir=output_dir, adult_filter_off=True, 
                            force_replace=False, timeout=60, verbose=True)
    def gather_images(self):
        # method gatheres all the needed images
        for category in self.categories:
            self.scrap_images(query=category, limit=self.num_imgs, output_dir=self.output_dir)
    
    def single_image_processing(self, filename, selected_categories=[0]):
        # method performs detection on an image (filename.jpg) and annotates it (filename.txt)
        img = self.open_img(path=filename) # open image
        result = self.predict_single_image(img=img) # object detection
        self.annotate_single_image(filename=filename, 
                                   result=result, 
                                   selected_categories=selected_categories) # annotaing file

    def open_img(self, path):
        #open image file
        return cv2.imread(path)

    def predict_single_image(self, img):
        return self.yolo_model.predict(img)
    
    def annotate_single_image(self, filename, result, selected_categories):
        filename = filename.split('.')[0]
        filename += '.txt'
        label_file = open(file=filename, mode="w+")
        for box in result[0].boxes: # loop over detected objects' bboxes
            cls = box.cls.cpu().numpy().astype(int)[0]
            xywhn = box.xywhn.cpu().numpy().astype(float)[0]
            label = f'{cls} {xywhn[0]:0.6f} {xywhn[1]:0.6f} {xywhn[2]:0.6f} {xywhn[3]:0.6f} '
            if cls in selected_categories:
                label_file.write(label)
                label_file.write('\n')
        label_file.close()
    
    def annotate_images(self):
        # method annotates images in their subdirs
        img_dir = self.original_cwd + "\\" + self.output_dir # path of images top dir
        img_subdirs = [x[0] for x in os.walk(img_dir)][1:] # list containing subdirs, '0'th element is the top dir, so eliminate it
        for img_subdir in img_subdirs: # loop over subdirs
            file_list = self.get_files_in_dir(path=img_subdir, filetype=".jpg") # get files in subdir
            for img_file in file_list: # loop over files
                img_file_path = img_subdir + "\\" + img_file
                self.single_image_processing(img_file_path) # annotate
    
    def get_files_in_dir(self, path, filetype=".jpg"):
        # get list of files in dir with a given extension
        files = [f for f in os.listdir(path) if f.endswith(filetype)]
        return files
    
    def change_class_all_files(self):
        # method changes classes i annot files in their subdirs
        img_dir = self.original_cwd + "\\" + self.output_dir # path of images top dir
        img_subdirs = [x[0] for x in os.walk(img_dir)][1:] # list containing subdirs, '0'th element is the top dir, so eliminate it
        category_index=0 # in ech subdir we will have distinct category index
        for img_subdir in img_subdirs: # loop over subdirs
            file_list = self.get_files_in_dir(path=img_subdir, filetype=".txt") # get annot files in subdir
            for img_file in file_list: # loop over files
                img_file_path = img_subdir + "\\" + img_file
                self.change_class_index(path=img_file_path, new_class=category_index)
            category_index += 1 # update index


    def change_class_index(self, path, new_class):
        # method changes class index in annot *.txt file
        f = open(file=path, mode="r+")
        lines = f.readlines()
        new_lines=[]
        for line in lines:
            new_lines.append(str(new_class)+line[1:])
        f.truncate(0)
        f.writelines(new_lines)
        f.close()
        # remove blank first line
        with open(path,'r') as fin:
            data = fin.read().splitlines(True)
        with open(path, 'w') as fout:
            fout.writelines(data[1:])

    def rename_images(self):
        # method renames images (in each sub dir index starts from 1, and I need a global index)

        img_dir = self.original_cwd + "\\" + self.output_dir # path of images top dir
        img_subdirs = [x[0] for x in os.walk(img_dir)][1:] # list containing subdirs, '0'th element is the top dir, so eliminate it
        for img_subdir in img_subdirs: # loop over subdirs
            file_list = self.get_files_in_dir(path=img_subdir, filetype=".jpg") # get annot files in subdir
            for img_file in file_list: # loop over files
                img_file_path = img_subdir + "\\" + img_file
                new_img_file = img_file.split('_')[0] +'_' + str(self.global_index) + '.jpg'
                new_img_file_path = img_subdir + '\\' + new_img_file
                os.rename(img_file_path,new_img_file_path)
                self.global_index += 1
    
    def prepare_dataset_dir_structure(self):
        # method prepares dir structure 
        if not os.path.exists(self.custom_dir):
            os.mkdir(self.custom_dir)
            os.mkdir(self.custom_dir + "\\images")
            os.mkdir(self.custom_dir + "\\images\\train")
            os.mkdir(self.custom_dir + "\\images\\val")
            os.mkdir(self.custom_dir + "\\labels")
            os.mkdir(self.custom_dir + "\\labels\\train")
            os.mkdir(self.custom_dir + "\\labels\\val")

    def prepare_yaml_file(self, yaml_filename="dataset.yaml"):
        # method prepares yaml file for YOLO8
        yaml_file = []
        yaml_file.append("# Train/val sets as path/to/imgs \n")
        yaml_file.append("path: ../" +self.custom_dir+" # dataset root dir\n")
        yaml_file.append("train: images/train # train images (relative to 'path')\n")
        yaml_file.append("val: images/val # val images (relative to 'path')\n")
        yaml_file.append("          \n")
        yaml_file.append("# Classes ("+str(len(self.categories))+" classes)\n")
        yaml_file.append("names:\n")
        for i in range(len(self.categories)):
            yaml_file.append("    "+str(i)+": "+self.categories[i]+"\n")
        with open(yaml_filename, 'w') as fout:
            fout.writelines(yaml_file)
        print(os.getcwd())
    
    def copy_files_to_dirs(self):
        # method copies files to respective dirs - prepare YOLO dataset
        img_dir = self.original_cwd + "\\" + self.output_dir # path of images top dir
        img_subdirs = [x[0] for x in os.walk(img_dir)][1:] # list containing subdirs, '0'th element is the top dir, so eliminate it
        
        img_dest_train = self.original_cwd + "\\" + self.custom_dir + "\\images\\train" # destination dir
        img_dest_val = self.original_cwd + "\\" + self.custom_dir + "\\images\\val"

        lbl_dest_train = self.original_cwd + "\\" + self.custom_dir + "\\labels\\train" # destination dir
        lbl_dest_val = self.original_cwd + "\\" + self.custom_dir + "\\labels\\val"
        
        
        train_val_switch = int(self.num_imgs * self.train_val_split_ratio) - 1 # index to switch destination dir from train to val
        
        for img_subdir in img_subdirs: # loop over subdirs
            file_list = self.get_files_in_dir(path=img_subdir, filetype=".jpg") # get files in subdir
            
            img_dest_dir = img_dest_train # initial value
            lbl_dest_dir = lbl_dest_train 
            train_val_idx = 0 # monitor number of files
            for img_file in file_list: # loop over files
                img_file_path = img_subdir + "\\" + img_file # source image path
                lbl_file = img_file.split(".")[0] + ".txt"
                lbl_file_path = img_subdir + "\\" + lbl_file
                if train_val_idx > train_val_switch:
                    img_dest_dir = img_dest_val
                    lbl_dest_dir = lbl_dest_val
                shutil.copy(img_file_path, img_dest_dir) # copy into dest dir
                shutil.copy(lbl_file_path, lbl_dest_dir) # copy into dest dir
                train_val_idx += 1
                


        


    ######################################################
    ######## ponizej testowe metody ######################
    ######################################################
    def annotated_bbox(self, frame, box, cls):
        # draw annotated bounding box
        color = (255, 0, 255) # purple
        white = (255, 255, 255)
        label = f"  {cls}" # custom label
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1) # draw bbox    
        
        # put in labels
        if box[1] - 25 > 0: # if the bbox upper edge allows label to be read
            cv2.rectangle(frame, (box[0], box[1]-30), (box[2], box[1]), color, -1) #    
            cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2) # annotate
        else: # if not, draw it in the bottom
            cv2.rectangle(frame, (box[0], box[3]-30), (box[2], box[3]), color, -1) #    
            cv2.putText(frame, label, (box[0], box[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2) # annotate
        return frame
    
    def plot_bboxes(self, results, frame):
        #plots bboxes on image
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        for box, score, cls  in zip(boxes, scores, classes):
            if score > 0.3:
                frame = self.annotated_bbox(frame, box, cls)
        return frame


    



yd = YoloDataset()
yd.prepare_dataset_dir_structure()
#yd.prepare_yaml_file()
#yd.gather_images()
#yd.rename_images()

#yd.annotate_images()
#yd.change_class_all_files()

yd.copy_files_to_dirs()


'''
img = yd.open_img("Image_8.jpg")
result = yd.predict_single_image(img)


frame = yd.plot_bboxes(result, img)

cv2.imshow("img", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''