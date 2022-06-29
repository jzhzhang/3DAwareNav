import copy


scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Hanson',
    'Hiteman',
    'Klickitat',
    'Lakeville',
    'Leonardo',
    'Lindenwood',
    'Marstons',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Tolstoy',
    'Wainscott',
    'Woodbine',
]

# scenes["train"] = [
#     'Beechwood',
#     'Allensville',
#     'Benevolence',
#     'Coffeen',
#     'Cosmos',
#     'Forkland',
#     'Hanson',
#     'Hiteman',
#     'Klickitat',
#     'Lakeville',
#     'Leonardo',
#     'Lindenwood',
#     'Marstons',
#     'Merom',
#     'Mifflinburg',
#     'Newfields',
#     'Onaga',
#     'Pinesdale',
#     'Pomaria',
#     'Ranchester',
#     'Shelbyville',
#     'Stockman',
#     'Tolstoy',
#     'Wainscott',
#     'Woodbine',
# ]



scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}






coco_index_mapping_array = [56,
                            57,
                            58,
                            59,
                            61,
                            62,
                            60,
                            69,
                            71,
                            72,
                            73,
                            74,
                            75,
                            41,
                            39]


coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]


# {"info": {"description": "COCO 2018 Panoptic Fake Samples for Testing", "url": "", "version": "1.0.0", "year": 2019, "contributor": "TensorFlow Datasets Authors", "date_created": "2019-06-29"}, "licenses": [{"url": "", "id": 1, "name": "License_1"}, {"url": "", "id": 2, "name": "License_2"}, {"url": "", "id": 3, "name": "License_3"}, {"url": "", "id": 4, "name": "License_4"}, {"url": "", "id": 5, "name": "License_5"}, {"url": "", "id": 6, "name": "License_6"}, {"url": "", "id": 7, "name": "License_7"}, {"url": "", "id": 8, "name": "License_8"}], "images": [{"license": 4, "file_name": "000000397133.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg", "height": 427, "width": 640, "date_captured": "2013-11-14 17:02:52", "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg", "id": 397133}, {"license": 1, "file_name": "000000037777.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg", "height": 230, "width": 352, "date_captured": "2013-11-14 20:55:31", "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg", "id": 37777}, {"license": 4, "file_name": "000000252219.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000252219.jpg", "height": 428, "width": 640, "date_captured": "2013-11-14 22:32:02", "flickr_url": "http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg", "id": 252219}], "annotations": [{"segments_info": [{"area": 353, "category_id": 52, "iscrowd": 0, "id": 6202563, "bbox": [221, 179, 37, 27]}, {"area": 99, "category_id": 55, "iscrowd": 0, "id": 2793704, "bbox": [231, 178, 11, 11]}, {"area": 223, "category_id": 55, "iscrowd": 0, "id": 1872866, "bbox": [216, 185, 17, 16]}, {"area": 155, "category_id": 55, "iscrowd": 0, "id": 1811173, "bbox": [218, 201, 14, 13]}, {"area": 205, "category_id": 55, "iscrowd": 0, "id": 1415646, "bbox": [232, 201, 16, 16]}, {"area": 227, "category_id": 55, "iscrowd": 0, "id": 1545703, "bbox": [205, 187, 15, 21]}, {"area": 553, "category_id": 62, "iscrowd": 0, "id": 13160659, "bbox": [28, 215, 60, 15]}, {"area": 393, "category_id": 62, "iscrowd": 0, "id": 9937583, "bbox": [117, 190, 50, 25]}, {"area": 605, "category_id": 62, "iscrowd": 0, "id": 9214632, "bbox": [243, 180, 50, 46]}, {"area": 85, "category_id": 64, "iscrowd": 0, "id": 6324097, "bbox": [103, 119, 7, 17]}, {"area": 4815, "category_id": 67, "iscrowd": 0, "id": 3886471, "bbox": [86, 178, 201, 49]}, {"area": 3592, "category_id": 79, "iscrowd": 0, "id": 12172223, "bbox": [137, 124, 61, 71]}, {"area": 58, "category_id": 81, "iscrowd": 0, "id": 9805731, "bbox": [268, 134, 26, 4]}, {"area": 6991, "category_id": 82, "iscrowd": 0, "id": 13552068, "bbox": [302, 75, 49, 151]}, {"area": 258, "category_id": 100, "iscrowd": 0, "id": 7182527, "bbox": [337, 54, 15, 22]}, {"area": 3414, "category_id": 112, "iscrowd": 0, "id": 13159886, "bbox": [42, 81, 50, 91]}, {"area": 588, "category_id": 130, "iscrowd": 0, "id": 14082533, "bbox": [71, 0, 136, 112]}, {"area": 172, "category_id": 176, "iscrowd": 0, "id": 5661298, "bbox": [185, 114, 14, 21]}, {"area": 2304, "category_id": 181, "iscrowd": 0, "id": 11646652, "bbox": [241, 82, 63, 47]}, {"area": 19551, "category_id": 186, "iscrowd": 0, "id": 12369342, "bbox": [0, 0, 352, 79]}, {"area": 9041, "category_id": 188, "iscrowd": 0, "id": 2908831, "bbox": [97, 61, 246, 120]}, {"area": 632, "category_id": 189, "iscrowd": 0, "id": 4281750, "bbox": [79, 226, 212, 4]}, {"area": 8012, "category_id": 190, "iscrowd": 0, "id": 10333111, "bbox": [0, 161, 312, 69]}, {"area": 16843, "category_id": 199, "iscrowd": 0, "id": 10596797, "bbox": [0, 35, 352, 155]}], "file_name": "000000037777.png", "image_id": 37777}, {"segments_info": [{"area": 7772, "category_id": 1, "iscrowd": 0, "id": 8160405, "bbox": [10, 167, 122, 227]}, {"area": 11341, "category_id": 1, "iscrowd": 0, "id": 4996146, "bbox": [510, 171, 124, 216]}, {"area": 8384, "category_id": 1, "iscrowd": 0, "id": 6446170, "bbox": [326, 175, 72, 197]}, {"area": 3188, "category_id": 10, "iscrowd": 0, "id": 7893105, "bbox": [337, 44, 61, 57]}, {"area": 2944, "category_id": 28, "iscrowd": 0, "id": 3101277, "bbox": [561, 90, 79, 67]}, {"area": 611, "category_id": 31, "iscrowd": 0, "id": 4009761, "bbox": [46, 213, 34, 50]}, {"area": 140, "category_id": 47, "iscrowd": 0, "id": 8748406, "bbox": [345, 226, 11, 22]}, {"area": 35051, "category_id": 191, "iscrowd": 0, "id": 8947339, "bbox": [0, 325, 640, 103]}, {"area": 49423, "category_id": 197, "iscrowd": 0, "id": 4670528, "bbox": [0, 0, 640, 410]}, {"area": 109152, "category_id": 199, "iscrowd": 0, "id": 10320957, "bbox": [32, 0, 466, 367]}], "file_name": "000000252219.png", "image_id": 252219}, {"segments_info": [{"area": 17418, "category_id": 1, "iscrowd": 0, "id": 5264729, "bbox": [389, 70, 109, 277]}, {"area": 1045, "category_id": 1, "iscrowd": 0, "id": 5069172, "bbox": [0, 263, 62, 37]}, {"area": 1482, "category_id": 44, "iscrowd": 0, "id": 6184554, "bbox": [218, 241, 39, 57]}, {"area": 416, "category_id": 47, "iscrowd": 0, "id": 7106418, "bbox": [119, 273, 25, 34]}, {"area": 887, "category_id": 47, "iscrowd": 0, "id": 4212043, "bbox": [141, 268, 33, 36]}, {"area": 128, "category_id": 49, "iscrowd": 0, "id": 1582137, "bbox": [136, 249, 21, 29]}, {"area": 101, "category_id": 50, "iscrowd": 0, "id": 1583422, "bbox": [166, 256, 9, 19]}, {"area": 351, "category_id": 51, "iscrowd": 0, "id": 3358794, "bbox": [156, 169, 26, 17]}, {"area": 2135, "category_id": 51, "iscrowd": 0, "id": 9808051, "bbox": [31, 344, 68, 41]}, {"area": 1795, "category_id": 51, "iscrowd": 0, "id": 4152689, "bbox": [60, 287, 75, 42]}, {"area": 219, "category_id": 51, "iscrowd": 0, "id": 5200226, "bbox": [157, 114, 18, 16]}, {"area": 24, "category_id": 56, "iscrowd": 0, "id": 4349026, "bbox": [70, 296, 9, 5]}, {"area": 130, "category_id": 56, "iscrowd": 0, "id": 1847864, "bbox": [87, 294, 23, 11]}, {"area": 30, "category_id": 56, "iscrowd": 0, "id": 1782058, "bbox": [99, 305, 10, 5]}, {"area": 25, "category_id": 57, "iscrowd": 0, "id": 1390975, "bbox": [97, 297, 7, 5]}, {"area": 46120, "category_id": 67, "iscrowd": 0, "id": 4612219, "bbox": [1, 240, 347, 187]}, {"area": 7036, "category_id": 79, "iscrowd": 0, "id": 263429, "bbox": [0, 211, 191, 99]}, {"area": 10067, "category_id": 79, "iscrowd": 0, "id": 592654, "bbox": [1, 164, 192, 99]}, {"area": 2289, "category_id": 81, "iscrowd": 0, "id": 3159353, "bbox": [497, 203, 122, 29]}, {"area": 2603, "category_id": 130, "iscrowd": 0, "id": 4423074, "bbox": [182, 0, 366, 67]}, {"area": 5490, "category_id": 175, "iscrowd": 0, "id": 1713717, "bbox": [0, 127, 192, 77]}, {"area": 12977, "category_id": 184, "iscrowd": 0, "id": 2968142, "bbox": [0, 0, 310, 161]}, {"area": 3280, "category_id": 188, "iscrowd": 0, "id": 3823996, "bbox": [416, 219, 81, 108]}, {"area": 256, "category_id": 189, "iscrowd": 0, "id": 3099756, "bbox": [0, 240, 263, 187]}, {"area": 28614, "category_id": 190, "iscrowd": 0, "id": 2108735, "bbox": [292, 311, 348, 116]}, {"area": 691, "category_id": 191, "iscrowd": 0, "id": 1777705, "bbox": [344, 309, 46, 22]}, {"area": 196, "category_id": 196, "iscrowd": 0, "id": 5665922, "bbox": [0, 288, 17, 64]}, {"area": 43041, "category_id": 199, "iscrowd": 0, "id": 4940924, "bbox": [157, 0, 474, 242]}], "file_name": "000000397133.png", "image_id": 397133}], "categories": [{"supercategory": "person", "isthing": 1, "id": 1, "name": "person"}, {"supercategory": "vehicle", "isthing": 1, "id": 2, "name": "bicycle"}, {"supercategory": "vehicle", "isthing": 1, "id": 3, "name": "car"}, {"supercategory": "vehicle", "isthing": 1, "id": 4, "name": "motorcycle"}, {"supercategory": "vehicle", "isthing": 1, "id": 5, "name": "airplane"}, {"supercategory": "vehicle", "isthing": 1, "id": 6, "name": "bus"}, {"supercategory": "vehicle", "isthing": 1, "id": 7, "name": "train"}, {"supercategory": "vehicle", "isthing": 1, "id": 8, "name": "truck"}, {"supercategory": "vehicle", "isthing": 1, "id": 9, "name": "boat"}, {"supercategory": "outdoor", "isthing": 1, "id": 10, "name": "traffic light"}, {"supercategory": "outdoor", "isthing": 1, "id": 11, "name": "fire hydrant"}, {"supercategory": "outdoor", "isthing": 1, "id": 13, "name": "stop sign"}, {"supercategory": "outdoor", "isthing": 1, "id": 14, "name": "parking meter"}, {"supercategory": "outdoor", "isthing": 1, "id": 15, "name": "bench"}, {"supercategory": "animal", "isthing": 1, "id": 16, "name": "bird"}, {"supercategory": "animal", "isthing": 1, "id": 17, "name": "cat"}, {"supercategory": "animal", "isthing": 1, "id": 18, "name": "dog"}, {"supercategory": "animal", "isthing": 1, "id": 19, "name": "horse"}, {"supercategory": "animal", "isthing": 1, "id": 20, "name": "sheep"}, {"supercategory": "animal", "isthing": 1, "id": 21, "name": "cow"}, {"supercategory": "animal", "isthing": 1, "id": 22, "name": "elephant"}, {"supercategory": "animal", "isthing": 1, "id": 23, "name": "bear"}, {"supercategory": "animal", "isthing": 1, "id": 24, "name": "zebra"}, {"supercategory": "animal", "isthing": 1, "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "isthing": 1, "id": 27, "name": "backpack"}, {"supercategory": "accessory", "isthing": 1, "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "isthing": 1, "id": 31, "name": "handbag"}, {"supercategory": "accessory", "isthing": 1, "id": 32, "name": "tie"}, {"supercategory": "accessory", "isthing": 1, "id": 33, "name": "suitcase"}, {"supercategory": "sports", "isthing": 1, "id": 34, "name": "frisbee"}, {"supercategory": "sports", "isthing": 1, "id": 35, "name": "skis"}, {"supercategory": "sports", "isthing": 1, "id": 36, "name": "snowboard"}, {"supercategory": "sports", "isthing": 1, "id": 37, "name": "sports ball"}, {"supercategory": "sports", "isthing": 1, "id": 38, "name": "kite"}, {"supercategory": "sports", "isthing": 1, "id": 39, "name": "baseball bat"}, 
# {"supercategory": "sports", "isthing": 1, "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "isthing": 1, "id": 41, "name": "skateboard"}, {"supercategory": "sports", "isthing": 1, "id": 42, "name": "surfboard"}, {"supercategory": "sports", "isthing": 1, "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "isthing": 1, "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "isthing": 1, "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "isthing": 1, "id": 47, "name": "cup"}, {"supercategory": "kitchen", "isthing": 1, "id": 48, "name": "fork"}, {"supercategory": "kitchen", "isthing": 1, "id": 49, "name": "knife"}, {"supercategory": "kitchen", "isthing": 1, "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "isthing": 1, "id": 51, "name": "bowl"}, {"supercategory": "food", "isthing": 1, "id": 52, "name": "banana"}, {"supercategory": "food", "isthing": 1, "id": 53, "name": "apple"}, {"supercategory": "food", "isthing": 1, "id": 54, "name": "sandwich"}, {"supercategory": "food", "isthing": 1, "id": 55, "name": "orange"}, {"supercategory": "food", "isthing": 1, "id": 56, "name": "broccoli"}, {"supercategory": "food", "isthing": 1, "id": 57, "name": "carrot"}, {"supercategory": "food", "isthing": 1, "id": 58, "name": "hot dog"}, {"supercategory": "food", "isthing": 1, "id": 59, "name": "pizza"}, {"supercategory": "food", "isthing": 1, "id": 60, "name": "donut"}, {"supercategory": "food", "isthing": 1, "id": 61, "name": "cake"}, {"supercategory": "furniture", "isthing": 1, "id": 62, "name": "chair"}, {"supercategory": "furniture", "isthing": 1, "id": 63, "name": "couch"}, 
# {"supercategory": "furniture", "isthing": 1, "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "isthing": 1, "id": 65, "name": "bed"}, {"supercategory": "furniture", "isthing": 1, "id": 67, "name": "dining table"}, {"supercategory": "furniture", "isthing": 1, "id": 70, "name": "toilet"}, {"supercategory": "electronic", "isthing": 1, "id": 72, "name": "tv"}, {"supercategory": "electronic", "isthing": 1, "id": 73, "name": "laptop"}, {"supercategory": "electronic", "isthing": 1, "id": 74, "name": "mouse"}, {"supercategory": "electronic", "isthing": 1, "id": 75, "name": "remote"}, {"supercategory": "electronic", "isthing": 1, "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "isthing": 1, "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "isthing": 1, "id": 78, "name": "microwave"}, {"supercategory": "appliance", "isthing": 1, "id": 79, "name": "oven"}, {"supercategory": "appliance", "isthing": 1, "id": 80, "name": "toaster"}, {"supercategory": "appliance", "isthing": 1, "id": 81, "name": "sink"}, {"supercategory": "appliance", "isthing": 1, "id": 82, "name": "refrigerator"}, {"supercategory": "indoor", "isthing": 1, "id": 84, "name": "book"}, {"supercategory": "indoor", "isthing": 1, "id": 85, "name": "clock"}, {"supercategory": "indoor", "isthing": 1, "id": 86, "name": "vase"}, {"supercategory": "indoor", "isthing": 1, "id": 87, "name": "scissors"}, {"supercategory": "indoor", "isthing": 1, "id": 88, "name": "teddy bear"}, {"supercategory": "indoor", "isthing": 1, "id": 89, "name": "hair drier"}, {"supercategory": "indoor", "isthing": 1, "id": 90, "name": "toothbrush"}, {"supercategory": "textile", "isthing": 0, "id": 92, "name": "banner"}, {"supercategory": "textile", "isthing": 0, "id": 93, "name": "blanket"}, {"supercategory": "building", "isthing": 0, "id": 95, "name": "bridge"}, {"supercategory": "raw-material", "isthing": 0, "id": 100, "name": "cardboard"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 107, "name": "counter"}, {"supercategory": "textile", "isthing": 0, "id": 109, "name": "curtain"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 112, "name": "door-stuff"}, {"supercategory": "floor", "isthing": 0, "id": 118, "name": "floor-wood"}, {"supercategory": "plant", "isthing": 0, "id": 119, "name": "flower"}, {"supercategory": "food-stuff", "isthing": 0, "id": 122, "name": "fruit"}, {"supercategory": "ground", "isthing": 0, "id": 125, "name": "gravel"}, {"supercategory": "building", "isthing": 0, "id": 128, "name": "house"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 130, "name": "light"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 133, "name": "mirror-stuff"}, {"supercategory": "structural", "isthing": 0, "id": 138, "name": "net"}, {"supercategory": "textile", "isthing": 0, "id": 141, "name": "pillow"}, {"supercategory": "ground", "isthing": 0, "id": 144, "name": "platform"}, {"supercategory": "ground", "isthing": 0, "id": 145, "name": "playingfield"}, {"supercategory": "ground", "isthing": 0, "id": 147, "name": "railroad"}, {"supercategory": "water", "isthing": 0, "id": 148, "name": "river"}, {"supercategory": "ground", "isthing": 0, "id": 149, "name": "road"}, {"supercategory": "building", "isthing": 0, "id": 151, "name": "roof"}, {"supercategory": "ground", "isthing": 0, "id": 154, "name": "sand"}, {"supercategory": "water", "isthing": 0, "id": 155, "name": "sea"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 156, "name": "shelf"}, {"supercategory": "ground", "isthing": 0, "id": 159, "name": "snow"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 161, "name": "stairs"}, {"supercategory": "building", "isthing": 0, "id": 166, "name": "tent"}, {"supercategory": "textile", "isthing": 0, "id": 168, "name": "towel"}, {"supercategory": "wall", "isthing": 0, "id": 171, "name": "wall-brick"}, {"supercategory": "wall", "isthing": 0, "id": 175, "name": "wall-stone"}, {"supercategory": "wall", "isthing": 0, "id": 176, "name": "wall-tile"}, {"supercategory": "wall", "isthing": 0, "id": 177, "name": "wall-wood"}, {"supercategory": "water", "isthing": 0, "id": 178, "name": "water-other"}, {"supercategory": "window", "isthing": 0, "id": 180, "name": "window-blind"}, {"supercategory": "window", "isthing": 0, "id": 181, "name": "window-other"}, {"supercategory": "plant", "isthing": 0, "id": 184, "name": "tree-merged"}, {"supercategory": "structural", "isthing": 0, "id": 185, "name": "fence-merged"}, {"supercategory": "ceiling", "isthing": 0, "id": 186, "name": "ceiling-merged"}, {"supercategory": "sky", "isthing": 0, "id": 187, "name": "sky-other-merged"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 188, "name": "cabinet-merged"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 189, "name": "table-merged"}, {"supercategory": "floor", "isthing": 0, "id": 190, "name": "floor-other-merged"}, {"supercategory": "ground", "isthing": 0, "id": 191, "name": "pavement-merged"}, {"supercategory": "solid", "isthing": 0, "id": 192, "name": "mountain-merged"}, {"supercategory": "plant", "isthing": 0, "id": 193, "name": "grass-merged"}, {"supercategory": "ground", "isthing": 0, "id": 194, "name": "dirt-merged"}, {"supercategory": "raw-material", "isthing": 0, "id": 195, "name": "paper-merged"}, {"supercategory": "food-stuff", "isthing": 0, "id": 196, "name": "food-other-merged"}, {"supercategory": "building", "isthing": 0, "id": 197, "name": "building-other-merged"}, {"supercategory": "solid", "isthing": 0, "id": 198, "name": "rock-merged"}, {"supercategory": "wall", "isthing": 0, "id": 199, "name": "wall-other-merged"}, {"supercategory": "textile", "isthing": 0, "id": 200, "name": "rug-merged"}]}


mpcat40_labels = [
    # '', # -1
    #'void', # 0
    'wall',
    'floor',
    'chair',
    'door',
    'table', # 5
    'picture',
    'cabinet',
    'cushion',
    'window',
    'sofa', # 10
    'bed',
    'curtain',
    'chest_of_drawers',
    'plant',
    'sink',
    'stairs',
    'ceiling',
    'toilet',
    'stool',
    'towel', # 20
    'mirror',
    'tv_monitor',
    'shower',
    'column',
    'bathtub',
    'counter',
    'fireplace',
    'lighting',
    'beam',
    'railing',
    'shelving',
    'blinds',
    'gym_equipment', # 33
    'seating',
    'board_panel',
    'furniture',
    'appliances',
    'clothes',
    'objects',
    'misc',
    'unlabeled' # 41
]


# habitat_labels = {
#             # 'background': 0,
#             'chair': 0, #g
#             'bed': 1, #g
#             'plant':2, #b
#             'toilet':3, # in resnet
#             'tv_monitor':4, # in resnet
#             'sofa':5, #g
#             'cabinet':6, #g
#             'chest_of_drawers':7, #b in resnet
#             'picture':8, #g
#             'sink':9, #g
#             'cushion':10, #g
#             'stool':11, #b
#             'towel':12, #b in resnet
#             'table':13, #g
#             'shower':14, #b
#             'bathtub':15 #b in resnet
# }

# habitat_labels = {
#             # 'background': 0,
#             'chair': 0, #g
#             'bed': 1, #g
#             'plant':2, #b
#             'toilet':3, # in resnet
#             'tv_monitor':4, # in resnet
#             'sofa':5,
#             'cabinet':6, #g
# }


# HM_semantic_name_2_id = {
#     "chair": 0,
#     "bed": 1,
#     "plant": 2,
#     "toilet": 3,
#     "tv_monitor": 4,
#     "sofa": 5
# }

# HM_semantic_name_2_id = {
#     "chair": 0,
#     "bed": 1,
#     "plant": 2,
#     "toilet": 3,
#     "tv_monitor": 4,
#     "sofa": 5
# }



habitat_labels = {
            'chair': 0, #g
            'table': 1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'background': 21
}


fourty221_ori = {}
# twentyone240 = {}
for i in range(len(mpcat40_labels)):
    lb = mpcat40_labels[i]
    if lb in habitat_labels.keys():
        fourty221_ori[i] = habitat_labels[lb]
        # twentyone240[habitat_labels[lb]] = i
print("habitat_labels",habitat_labels)
print(fourty221_ori)
print(len(fourty221_ori))
fourty221 = copy.deepcopy(fourty221_ori)