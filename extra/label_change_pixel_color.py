import argparse
import os
from pathlib import Path
from tqdm import tqdm
import magic
from PIL import Image, ImageColor
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LabelChangePixelColor():

    def __init__(self, label_dir, label_output_dir) -> None:
        assert os.path.exists(label_dir), "{} don't exists".format(label_dir)

        self.label_dir = label_dir
        self.label_output_dir = label_output_dir

        Path(label_output_dir).mkdir(parents=True, exist_ok=True)
        assert not os.listdir(label_output_dir), "{} isn't empty".format(label_dir)

    def run(self, ori_color: dict, new_color: dict) -> None:
        progress = tqdm(total=self.__count_total(self.label_dir))
        lower = np.array(list(ori_color), dtype = "uint8") 
        upper = np.array(list(ori_color), dtype = "uint8")


        for root, _, files in os.walk(self.label_dir):
            for name in files:
                if self.__check_is_png(os.path.join(root, name)) is True:
                    file_path = os.path.join(root, name)
                    input_file = os.path.splitext(file_path)[0] + ".png"
                    
                    output_path = os.path.join(self.label_output_dir, name)

                    # Abrir a imagen original em formato RGB                    
                    image = Image.open(input_file)
                    image = image.convert('RGB')
                    pixels = image.load() # create the pixel map
                    
                    # Percorrer cada pixel da imagem
                    can_save = False
                    for i in range(image.size[0]): # for every pixel:
                        for j in range(image.size[1]):
                            # Converter cada pixel para a nova cor, caso necessário
                            if pixels[i,j] == ori_color:
                                can_save=True
                                pixels[i,j] = new_color

                    # Caso algum pixel seja modificado, salvar a imagem em uma nova pasta
                    if can_save:
                        image.save(output_path)
                    
                    # cv2_image = np.array(image)
                    # mask = cv2.inRange(cv2_image, lower, upper)
                    # detected_output = cv2.bitwise_and(cv2_image, cv2_image, mask =  mask) 
                    # plt.imshow(detected_output)
                    # plt.show()

    def __check_is_png(self, filepath: str) -> bool:
        allowed_types = [
            'image/png',
        ]

        if magic.from_file(filepath, mime=True) not in allowed_types:
            return False
        return True


    def __count_total(self, path: str) -> int:
        print('Please wait till total files are counted...')
        print(path)

        result = 0

        for root, _, files in os.walk(path):
            for name in files:
                if self.__check_is_png(os.path.join(root, name)) is True:
                    result += 1

        return result

# Função criada para percorrer uma pasta com imagens (labels RGB) e trocar uma cor por outra
# Ex: Trocar a cor azul (0,0,255) pela cor amarela (255,255,0) de todas as imagens em um diretório
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Pixel Color of Multiple Image Labels')
    parser.add_argument('--label_dir', type=str, help='Path do directory with label images')
    parser.add_argument('--label_output_dir', type=str, help='Path do output directory', default='D:\\datasets\\ICMBIO_NOVO\\all\label_changed')
    parser.add_argument('--ori_color', type=str, help='Original Pixel Color', default=(0, 255, 197))
    parser.add_argument('--new_color', type=str, help='New Pixel Color', default=(255, 0, 0))
    args = parser.parse_args()
    
    LabelChangePixelColor(args.label_dir, args.label_output_dir).run(args.ori_color, args.new_color)

    # Execute
    # python .\extra\label_change_pixel_color.py --label_dir='D:\datasets\ICMBIO_NOVO\all\label'