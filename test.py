import os
import shutil

# 대상 디렉토리
base_dir = "D:/Jiwoon/dataset/mvtec_anomaly_detection/"

# 각 하위 디렉토리에 대해
for sub_dir in os.listdir(base_dir):
    # test 폴더를 찾음
    test_dir = os.path.join(base_dir, sub_dir, 'test')
    if os.path.exists(test_dir):
        # bad, good 폴더 경로
        bad_dir = os.path.join(test_dir, 'bad')
        good_dir = os.path.join(test_dir, 'good')

        # vit_test 폴더 경로
        vit_test_dir = os.path.join(test_dir, 'vit_test')
        # vit_test 폴더가 없는 경우 생성
        if not os.path.exists(vit_test_dir):
            os.makedirs(vit_test_dir)

        # bad, good 폴더를 vit_test 폴더로 복사
        if os.path.exists(bad_dir):
            shutil.copytree(bad_dir, os.path.join(vit_test_dir, 'bad'))
        if os.path.exists(good_dir):
            shutil.copytree(good_dir, os.path.join(vit_test_dir, 'good'))


# # 복사할 원본 폴더의 경로와 복사할 대상 폴더의 경로를 지정합니다.
# base_dir = 'D:/Jiwoon/dataset/mvtec_anomaly_detection/'
#
# src_folder = "D:/Jiwoon/dataset/mvtec_anomaly_detection/bottle/test/"
# dst_folder = "D:/Jiwoon/dataset/mvtec_anomaly_detection/bottle/test/bad/"
#
# category = os.listdir(base_dir)
#
# for cate in category:
#
#     src_folder = os.path.join(base_dir, cate, 'test/')
#
#     src_folder_name = [folder for folder in os.listdir(src_folder) if folder != "good"]
#
#     dst_folder = os.path.join(base_dir, cate, 'test/bad/')
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#
#     for folder in src_folder_name:
#         folder_path = os.path.join(src_folder, folder)
#         if os.path.isdir(folder_path):
#             image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#             # print(f"Folder: {folder}")
#             # print(f"Image Files: {image_files}\n")
#
#             for image_file in image_files:
#                 # 대상 폴더명을 파일명 앞에 붙여 새로운 파일명을 만듭니다.
#                 new_name = folder + "_" + image_file
#
#                 # 원본 파일의 경로와 복사할 파일의 경로를 지정합니다.
#                 src_path = os.path.join(folder_path, image_file)
#                 dst_path = os.path.join(dst_folder, new_name)
#                 # 원본 파일을 새로운 파일명으로 복사합니다.
#                 shutil.copy2(src_path, dst_path)

