import logging
import os
import sys
import time
from PIL import Image
import sys
from paddleocr import PaddleOCR
import numpy as np


success_path = []
model_path = os.path.dirname(os.path.abspath(__file__)) + r'\\model\\'
PADDLE_OCR = PaddleOCR(use_gpu=False, det_model_dir=model_path + 'det', cls_model_dir=model_path + 'cls', rec_model_dir=model_path + 'rec', rec_char_dict_path=model_path + 'ppocr_keys_v1.txt')


def rotate_img(img_path, angle):
    im = Image.open(img_path)
    im_rotate = im.rotate(angle)
    # im_rotate.show()
    return np.array(im_rotate)[:, :, :3]


def img_to_text(img):
    for i in [0, 45, 90, 180]:
        img_path = rotate_img(img, i)
        # if len(sys.argv) > 1:
        #     img_path = r"{}".format(sys.argv[1])
        ocr_result = PADDLE_OCR.ocr(img_path, cls=True)
        ocr_result = [line[1][0] for line in ocr_result]
        # result = '\n'.join(ocr_result)
        result = ''.join(ocr_result)
        for key in ["秘密", "机密", "绝密"]:
            if key in result:
                success_path.append(img)
                return


if __name__ == '__main__':

    # 为了测试方便，这个文件夹下全是图片
    work_dir = r"E:\demo\DLAN_code\audit-client-pyqt\src\dlan\test\ocr_test\dist\Easy图片60"
    if len(sys.argv) > 1:
        work_dir = r"{}".format(sys.argv[1])

    start_time = time.time()

    for parent, dirnames, filenames in os.walk(work_dir):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            try:
                img_to_text(file_path)
            except:
                pass
    end_time = time.time()

    print("\n一共花了:{} 秒".format(end_time - start_time))
    print("包含关键字的图片有 {} 张".format(len(success_path)))
    print("包含关键字图片的路径：{} ".format(success_path))

