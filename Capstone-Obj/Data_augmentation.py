import os
import random
import numpy as np
import cv2
import glob
from PIL import Image
import PIL.ImageOps

# 현재 경로 표시
current_path = os.path.abspath(os.curdir)
print("Current path is {}".format(current_path))

current_path = '/Users/jongukyang/Test_OpenCV_DeepLearning/Test_Make_YOLO'
print(current_path)

# 원본 사진이 있는 폴더
file_path = '/h'
# 위의 폴더에 있는 이미지 이름의 배열 저장
path = current_path + file_path + '/'
print(path)
file_names = os.listdir(current_path + file_path)
print(file_names)
total_origin_image_num = len(file_names)
print(total_origin_image_num)

# 다운받을 절대주소
download_path = current_path + '/images2/'

# 몇번의 augmentation이 일어났는지 확인
augment_cnt = 1

# 다음 변수를 수정하여 새로 만들 이미지 갯수를 정합니다.
num_augmented_images = 50

for i in range(1, num_augmented_images):
    # 전체 이미지 개수 중 하나를 랜덤 선택, file_name에 그 인덱스를 갖는 이미지 이름 저장
    change_picture_index = random.randrange(1, total_origin_image_num - 1)
    print(change_picture_index)
    print(file_names[change_picture_index])
    file_name = file_names[change_picture_index]
    print(file_name)

    origin_image_path = path + file_name
    print(origin_image_path)
    image = Image.open(origin_image_path)
    random_augment = random.randrange(1, 4) # 무작위로 1~3의 변수 결정
    print(random_augment)

    if (random_augment == 1):
        # 이미지 좌우 반전
        print("invert")
        inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        inverted_image.show()
        inverted_image.save(download_path + 'inverted_' + str(augment_cnt) + '.jpg')

    elif (random_augment == 2):
        # 이미지 기울이기
        print("rotate")
        rotated_image = image.rotate(random.randrange(-20, 20))
        rotated_image.save(download_path + 'rotated_' + str(augment_cnt) + '.jpg')

    elif (random_augment == 3):
        # 노이즈 추가하기
        img = cv2.imread(origin_image_path)
        print("noise")
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_array = img + gauss
        noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
        noisy_image.save(download_path + 'noiseAdded_' + str(augment_cnt) + '.jpg')

    augment_cnt += 1