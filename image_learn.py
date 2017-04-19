#coding:utf-8
import numpy as np
import os
import sys
import collections

class FilterFile(object):

    def __init__(self, path):
        self.allfiles = os.listdir(path)
        self.label_file_map = collections.defaultdict(list)
        self.generate_label_file_map()
        self.label_vector_map = collections.defaultdict(list)

    def process_single_image_2_vector(self, filename):
        from PIL import Image 
        img = Image.open(filename)
        return_vector = np.zeros((1,68*68))
        if hasattr(img, 'width'):
            width, height = img.width, img.height
        else:
            width, height = img.size

        for x in range(width):
            for y in range(height):
                return_vector[0, 68*x+y] = (img.getpixel((x,y)))
        return return_vector
    
    def generate_label_file_map(self):
        for elt in self.allfiles:
            number_label = elt.split('.')[0].split('_')[0]
            self.label_file_map[number_label].append(elt)
            
    def calculate_two_vector_distance(self, vec1, vec2):
        return np.sqrt(np.sum(np.square(vec1-vec2)))
    
    def test_file(self):
        vec_list_001 = []
        vec_list_108 = []
        for elt in self.label_file_map['001']:
            vec_list_001.append(self.process_single_image_2_vector('BmpMoban/'+elt))
        vec_start = vec_list_001[0]

        for elt in self.label_file_map['108']:
            vec_list_108.append(self.process_single_image_2_vector('BmpMoban/'+elt))

        for i in range(len(vec_list_108)):
            print(self.calculate_two_vector_distance(vec_start, vec_list_108[i]))

"""
Testing code
"""
def main():
    ff = FilterFile('BmpMoban')
    ff.test_file()
    #print(ff.label_file_map.keys())
    #ff.process_single_image('BmpMoban/001_a69a0e5e37b0e08edfed28fdba51191c_0.jpgresize.jpg')

if __name__ == '__main__':
    main()
        
