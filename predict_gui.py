import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton,QFileDialog,QDesktopWidget
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predict System")
        self.setGeometry(100, 100, 400, 300)

        # 创建五个按钮
        self.btn_predict = QPushButton("ImagePredict", self)
        self.btn_video = QPushButton("VideoDetect", self)
        self.btn_fps = QPushButton("TestFPS", self)
        self.btn_dir_predict = QPushButton("DirectoryPredict", self)
        self.btn_heatmap = QPushButton("DrawHeatmap", self)

        # 设置按钮位置和大小
        # 设置按钮位置和大小
        self.btn_predict.setGeometry(50, 50, 200, 30)
        self.btn_video.setGeometry(50, 100, 200, 30)
        self.btn_fps.setGeometry(50, 150, 200, 30)
        self.btn_dir_predict.setGeometry(50, 200, 200, 30)
        self.btn_heatmap.setGeometry(50, 250, 200, 30)
        
        # 创建按钮并连接点击事件
        self.btn_predict.clicked.connect(self.predict_image)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_fps.clicked.connect(self.test_fps)
        self.btn_dir_predict.clicked.connect(self.predict_directory)
        self.btn_heatmap.clicked.connect(self.visualize_heatmap)
        
    def predict_image(self):
        '''
        1、如果想要进行检测完的图片的保存, 利用r_image.save("img.jpg")即可保存, 直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标, 可以进入yolo.detect_image函数, 在绘图部分读取top, left, bottom, right这四个值。
        3、如果想要利用预测框截取下目标, 可以进入yolo.detect_image函数, 在绘图部分利用获取到的top, left, bottom, right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字, 比如检测到的特定目标的数量, 可以进入yolo.detect_image函数, 在绘图部分对predicted_class进行判断,
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车, 然后记录数量即可。利用draw.text即可写字。
        '''
        options = QFileDialog.Options()
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if img_path:
            try:
                image = Image.open(img_path)
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
                print("Image prediction done!")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No image selected.")
                
        
    def detect_video(self):
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    def test_fps(self):
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    def predict_directory(self):
        import os
        from tqdm import tqdm
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if dir_path:
            img_names = os.listdir(dir_path)
            for img_name in img_names:
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_path, img_name)
                    image = Image.open(image_path)
                    r_image     = yolo.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                    print(f"Processed: {img_name}")
            print("Directory prediction done!")
        else:
            print("No directory selected.")


    def visualize_heatmap(self):
        options = QFileDialog.Options()
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if img_path:
            try:
                image = Image.open(img_path)
                yolo.detect_heatmap(image, heatmap_save_path)
                print("Heatmap visualization done!")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No image selected.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
