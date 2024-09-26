import os
import shutil
import random


def replace_images(source_folder, target_folder):
    # 获取源文件夹中的所有文件
    source_images = os.listdir(source_folder)

    for image_name in source_images:
        source_image_path = os.path.join(source_folder, image_name)
        target_image_path = os.path.join(target_folder, image_name)

        # 检查目标文件夹中是否存在同名文件
        if os.path.isfile(target_image_path):
            print(f"替换: {source_image_path} -> {target_image_path}")
            # 替换源文件夹中的图像为目标文件夹中的图像
            shutil.copy(target_image_path, source_image_path)
        else:
            print(f"{image_name} 在目标文件夹中不存在，跳过替换。")


def split_datasets(UAV_DIR):
    train_dir = os.path.join(UAV_DIR, 'train')
    test_dir = os.path.join(UAV_DIR, 'test')
    drone_train_dir = os.path.join(train_dir, 'drone')
    satellite_train_dir = os.path.join(train_dir, 'satellite')
    drone_test_dir = os.path.join(test_dir, 'drone')
    satellite_test_dir = os.path.join(test_dir, 'satellite')

    os.makedirs(drone_train_dir, exist_ok=True)
    os.makedirs(satellite_train_dir, exist_ok=True)
    os.makedirs(drone_test_dir, exist_ok=True)
    os.makedirs(satellite_test_dir, exist_ok=True)

    scene_dirs = [os.path.join(UAV_DIR, f'0{i}') for i in range(1, 10)]
    scene_dirs.append(os.path.join(UAV_DIR, "11"))

    for scene in scene_dirs:
        if os.path.exists(scene):
            drone_dir = os.path.join(scene, 'drone')
            satellite_dir = os.path.join(scene, 'satellite')

            if os.path.exists(drone_dir) and os.path.exists(satellite_dir):
                drone_images = os.listdir(drone_dir)
                satellite_images = os.listdir(satellite_dir)
                satellite_filenames_set = {os.path.splitext(os.path.basename(name))[0] for name in satellite_images}

                drone_images = [image for image in drone_images if
                                os.path.splitext(os.path.basename(image))[0] in satellite_filenames_set]
                # for path in drone_images:
                #     # 分割文件名和后缀
                #     filename, _ = os.path.splitext(path)  # 只保留文件名，不需要原后缀
                #     new_path = filename + '.tif'  # 新的文件路径
                #     if new_path not in satellite_images:
                #         drone_images.remove(path)

                num_drone = len(drone_images)

                train_drone_count = int(num_drone * 0.8)

                train_drone_images = random.sample(drone_images, train_drone_count)
                train_satellite_images = []
                # 遍历所有图像路径
                for path in train_drone_images:
                    # 分割文件名和后缀
                    filename, _ = os.path.splitext(path)  # 只保留文件名，不需要原后缀
                    new_path = filename + '.tif'  # 新的文件路径
                    train_satellite_images.append(new_path)

                for img in train_drone_images:
                    shutil.copy(os.path.join(drone_dir, img), os.path.join(drone_train_dir, img))
                for img in train_satellite_images:
                    shutil.copy(os.path.join(satellite_dir, img), os.path.join(satellite_train_dir, img))

                test_drone_images = list(set(drone_images) - set(train_drone_images))
                test_satellite_images = []
                # 遍历所有图像路径
                for path in test_drone_images:
                    # 分割文件名和后缀
                    filename, _ = os.path.splitext(path)  # 只保留文件名，不需要原后缀
                    new_path = filename + '.tif'  # 新的文件路径
                    test_satellite_images.append(new_path)

                for img in test_drone_images:
                    shutil.copy(os.path.join(drone_dir, img), os.path.join(drone_test_dir, img))
                for img in test_satellite_images:
                    shutil.copy(os.path.join(satellite_dir, img), os.path.join(satellite_test_dir, img))


def split_test(UAV_DIR):
    test_dir = os.path.join(UAV_DIR, 'test')

    drone_test_dir = os.path.join(test_dir, 'drone')
    satellite_test_dir = os.path.join(test_dir, 'satellite')

    os.makedirs(drone_test_dir, exist_ok=True)
    os.makedirs(satellite_test_dir, exist_ok=True)

    scene_dirs = [os.path.join(UAV_DIR, f'0{i}') for i in range(1, 10)]
    scene_dirs.append(os.path.join(UAV_DIR, "11"))

    for scene in scene_dirs:
        if os.path.exists(scene):
            drone_dir = os.path.join(scene, 'drone')

            if os.path.exists(drone_dir):
                drone_images = os.listdir(drone_dir)
                satellite_images = os.listdir(satellite_test_dir)
                satellite_filenames_set = {os.path.splitext(os.path.basename(name))[0] for name in satellite_images}

                drone_images = [image for image in drone_images if
                                os.path.splitext(os.path.basename(image))[0] in satellite_filenames_set]
                for img in drone_images:
                    shutil.copy(os.path.join(drone_dir, img), os.path.join(drone_test_dir, img))


def check(source_dir, target_dir):
    """检查原文件夹中的图像是否在目标文件夹中存在对应图像，不存在则删去"""
    # 获取源文件夹下所有图像文件列表
    source_files = os.listdir(source_dir)
    image_files = [f for f in source_files if f.endswith('.JPG')]  # 只获取jpg文件

    # 遍历源文件夹中的每个图像文件
    for image in image_files:
        # 获取图像的文件名前缀（不包括扩展名）
        image_prefix = os.path.splitext(image)[0]  # 使用os.path.splitext分割文件名和扩展名

        # 构建目标文件夹下对应文件的名称（假设扩展名为.tif）
        target_file = f"{image_prefix}.tif"

        # 检查目标文件夹中是否存在对应的文件
        if not os.path.isfile(os.path.join(target_dir, target_file)):
            # 如果不存在对应的文件，则在源文件夹中删除该图像文件
            source_image_path = os.path.join(source_dir, image)
            os.remove(source_image_path)  # 删除文件，注意需要确保没有其他程序正在使用该文件
            print(f"删除了文件 {source_image_path}")  # 可选：打印删除操作的日志信息


if __name__ == "__main__":
    # source_dir = r"I:\datasets\UAV-VisLoc\test\drone"
    # target_dir = r"I:\datasets\UAV-VisLoc\drone\satellite"
    # split_test(r"I:\datasets\UAV-VisLoc")
    # split_datasets(r"I:\datasets\UAV-VisLoc")
    # check(source_dir, target_dir)
    source_folder = r"I:\datasets\UAV-VisLoc\test\gallery_satellite"
    target_folder = r"I:\datasets\UAV-VisLoc\04\satellite"
    replace_images(source_folder, target_folder)
