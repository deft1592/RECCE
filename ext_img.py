import cv2
import os
import glob


from retinaface import RetinaFace


# 打开视频文件
video_folder = 'dataset\Celeb-DF-v2\YouTube-real'
video_files = glob.glob(os.path.join(video_folder, '*.mp4')) + glob.glob(os.path.join(video_folder, '*.avi'))
if not os.path.exists(os.path.join(video_folder,'images')):
    os.mkdir(os.path.join(video_folder,'images'))

for video_path in video_files:
    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)

    # 读取视频帧并检测人脸
    frame_idx = 0
    save_frame_idx = 0  # 初始化保存帧的编号

    while cap.isOpened():
        # 每30帧处理一次
        if frame_idx % 30 == 0:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用Retinaface检测人脸
            faces = RetinaFace.detect_faces(frame)
        
            # 遍历检测到的人脸并保存
            for  face in faces.values():
                # 提取人脸区域
                x1, y1, x2, y2 = face['facial_area']
                face_img = frame[y1:y2, x1:x2]
                video_name=os.path.basename(video_path).replace('.','_')
                # 构建保存人脸的文件名
                face_filename =os.path.join(video_folder,'images',video_name + f'face_{save_frame_idx}.png')

                # 保存人脸图像
                cv2.imwrite(face_filename, face_img)
                print(f'Saved {face_filename}')

            # 更新保存帧的编号
            save_frame_idx += 1
        else:
            # 读取下一帧但不保存
            cap.read()

        # 更新帧编号
        frame_idx += 1
