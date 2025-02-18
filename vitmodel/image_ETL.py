import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2

"""
主要功能：
1. 从rsna-breast-cancer-detection-png256nearest文件夹中读取每一个png文件
2. 转换为三维数组
3. 根据train.csv中的image名字和对应的'cancer'是否为1将其存放到相应的文件夹内，如果为1，则放到positive文件夹内，如果为0，则放到nagetive文件夹内
4. 用Parallel并行处理图像
"""

def process_image(image_path):
    """
    处理单个图像，将其转换为三维数组。
    """
    try:
        image = Image.open(image_path)
        image_np = np.array(image)
        #print(f"Processing {image_path}: shape {image_np.shape}")
        
        # 将图像转换为三维数组 (256, 256, 1)
        if len(image_np.shape) == 2:
            image_np = image_np[:, :, np.newaxis]

        return image_np
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save_image(image_np, output_path):
    """
    将 numpy 数组转换回图像并保存。
    """
    try:
        print(image_np.shape)
        cv2.imwrite(output_path, image_np)
        #print(f"Saved image to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

def process_and_save_image(image_path, df, positive_folder, negative_folder):
    """
    处理图像并根据标签将其保存到相应的文件夹。
    """
    image_np = process_image(image_path)
    if image_np is None:
        return
    
    # 获取图像对应的标签
    image_file = os.path.basename(image_path)
    image_id = os.path.splitext(image_file)[0]
    try:
        label = df.loc[df['image_id'] == image_id, 'cancer'].values[0]
    except IndexError:
        print(f"Image ID {image_id} not found in CSV")
        return
    
    # 保存图像到相应的文件夹
    if label == 1:
        output_path = os.path.join(positive_folder, image_file)
    else:
        output_path = os.path.join(negative_folder, image_file)
    
    save_image(image_np, output_path)

def process_images_from_folder(folder_path, csv_path, output_folder, num_workers):
    """
    从指定文件夹中读取所有 PNG 图像，并根据 train.csv 中的标签将其存放到相应的文件夹内。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    def get_image_id(row):
        return str(row['patient_id']) + "_" + str(row['image_id'])
    df['image_id'] = df.apply(get_image_id, axis=1)
    
    # 创建输出文件夹
    positive_folder = os.path.join(output_folder, "positive")
    negative_folder = os.path.join(output_folder, "negative")
    os.makedirs(positive_folder, exist_ok=True)
    os.makedirs(negative_folder, exist_ok=True)
    
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]
    #print(image_paths)
    #print(f"Found {len(image_paths)} images in {folder_path}")
    
    # 并行处理图像
    _ = Parallel(n_jobs=num_workers)(
        delayed(process_and_save_image)(image_path, df, positive_folder, negative_folder)
        for image_path in tqdm(image_paths)
    )

"""
将图像裁剪成只含有有效数据的部分。
"""
def crop_and_resize(image_np, target_size=224):
    """
    对图像进行裁剪，只保留有效部分。
    """

    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    X = image_np[5:-5, 5:-5]

    # regions of non-empty pixels
    output = cv2.connectedComponentsWithStats((X > 20).astype(np.uint8)[:, :, 0], 8, cv2.CV_32S)

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]

    # finding max area which always corresponds to the breast data.
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h

    # cutting out the breast data
    X_fit = X[y1: y2, x1: x2]

    # 将裁剪后的图像转换为 PIL 图像
    image_pil = Image.fromarray(X_fit)

    # 将图像的宽度调整为 target_size
    width, height = image_pil.size
    new_width = target_size
    new_height = int(height * (target_size / width))
    resized_image = image_pil.resize((new_width, new_height), Image.BILINEAR)

    # 如果高度大于 target_size，则裁剪高度
    if new_height > target_size:
        top = (new_height - target_size) // 2
        bottom = top + target_size
        cropped_image = resized_image.crop((0, top, new_width, bottom))
    else:
        # 如果高度小于 target_size，则填充高度
        padding = (0, (target_size - new_height) // 2, 0, (target_size - new_height + 1) // 2)
        cropped_image = ImageOps.expand(resized_image, padding)

    return np.array(cropped_image)

def process_and_save_cropped_images(input_folder, output_folder, num_workers):
    """
    从指定文件夹中读取所有 PNG 图像，裁剪有效部分并保存到输出文件夹。
    """
    # 获取所有图像路径
    positive_folder = os.path.join(input_folder, "positive")
    negative_folder = os.path.join(input_folder, "negative")
    positive_image_paths = [os.path.join(positive_folder, filename) for filename in os.listdir(positive_folder) if filename.endswith(".png")]
    negative_image_paths = [os.path.join(negative_folder, filename) for filename in os.listdir(negative_folder) if filename.endswith(".png")]
    image_paths = positive_image_paths + negative_image_paths

    # 创建输出文件夹
    positive_output_folder = os.path.join(output_folder, "positive")
    negative_output_folder = os.path.join(output_folder, "negative")
    os.makedirs(positive_output_folder, exist_ok=True)
    os.makedirs(negative_output_folder, exist_ok=True)

    def process_and_save_image(image_path):
        """
        处理图像并保存裁剪后的图像。
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image, dtype=np.uint8)
            # 将图像转换为三维数组 (256, 256, 1)
            if len(image_np.shape) == 2:
                image_np = image_np[:, :, np.newaxis]

            # 对图像进行裁剪
            image_np = crop_and_resize(image_np)

            # 保存裁剪后的图像
            image_file = os.path.basename(image_path)
            if "positive" in image_path:
                output_path = os.path.join(positive_output_folder, image_file)
            else:
                output_path = os.path.join(negative_output_folder, image_file)
            cv2.imwrite(output_path, image_np)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # 并行处理图像
    _ = Parallel(n_jobs=num_workers)(
        delayed(process_and_save_image)(image_path)
        for image_path in tqdm(image_paths))


if __name__ == "__main__":
    folder_path = "../data/rsna-breast-cancer-detection-png256nearest/test"
    csv_path = "../data/train.csv"
    output_folder = "../data/xray/test"
    num_workers = 8  # 设置并行处理的工作线程数
    #process_images_from_folder(folder_path, csv_path, output_folder, num_workers)
    process_and_save_cropped_images("../data/small_xray", 
                                       "../data/small_xray_crop", 
                                       num_workers)