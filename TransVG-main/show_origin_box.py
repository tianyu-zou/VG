import json
import os
import cv2
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt


# 模拟你的 JSON 数据
data = [
    {
        "image_path": "images\\24_0_106_0000010_250128.jpg",
        "question": "A turned-off TV",
        "result": [
            [
                134.0,
                89.0
            ],
            [
                44.0,
                52.0
            ]
        ]
    },
    {
        "image_path": "images\\24_0_106_0000010_250128.jpg",
        "question": "A blue chair",
        "result": [
            [
                44.0,
                138.0
            ],
            [
                62.0,
                80.0
            ]
        ]
    },
    {
        "image_path": "images\\24_0_106_0000010_250128.jpg",
        "question": "A red sofa",
        "result": [
            [
                57.0,
                125.0
            ],
            [
                76.0,
                74.0
            ]
        ]
    }
]
# imsize是当前模型的imsize,VLTVG=TransVG=640,ClIPVG=224,EEVG=448
imsize = 224
# 输出目录
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 处理每条数据
images_cache = {}  # 缓存图片避免重复读取

for item in data:
    image_path = item["image_path"].replace("\\", "/")  # 替换为兼容路径
    image_path = osp.join('../VG-RS-images', os.path.basename(image_path.replace('\\', '/')))
    print(image_path)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, image_name)

    # 读取或复用图片
    if image_path not in images_cache:
        img = cv2.imread(image_path)
        if img is None:
            print(f"图像读取失败: {image_path}")
            continue
        images_cache[image_path] = img.copy()
    img = images_cache[image_path]
    print(img.shape)
    ratio = float(imsize / float(max(img.shape[0], img.shape[1])))
    # 获取坐标
    (x1, y1), (w, h) = item["result"]

    new_w, new_h = round(img.shape[0] * ratio), round(img.shape[1] * ratio)
    dw = imsize - new_w
    dh = imsize - new_h
    top = round(dh / 2.0 - 0.1)
    left = round(dw / 2.0 - 0.1)

    ori_x1 = (x1 - top) / ratio
    ori_y1 = (y1 - left) / ratio
    ori_w = w / ratio
    ori_h = h / ratio

    final_x1, final_y1, final_x2, final_y2 = (ori_x1 ), (ori_y1), (ori_x1 + ori_w), (ori_y1 + ori_h)

    x1, y1, x2, y2 = map(int, [ final_x1, final_y1, final_x2, final_y2])
    print(x1, y1, x2, y2)
    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 添加标签（问题）
    cv2.putText(img, item["question"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 保存图片
    cv2.imwrite(save_path, img)
    # plt.gca().add_patch(
    #     plt.Rectangle((int(ori_x1), int(ori_y1)), int(ori_w), int(ori_h),
    #                   color='g', fill=False, linewidth=4))

    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去刻度
    # plt.yticks([])
    # plt.imshow(img)  # 绘制图像，将CV的BGR换成RGB
    # # plt.show()  # 显示图像
    # # text_query = data_loader_test.dataset.images[_][2]
    # plt.savefig(save_path, dpi=800,
    #             bbox_inches='tight', pad_inches=0)
    # plt.clf()

print(f"所有图像已保存到: {output_dir}")
