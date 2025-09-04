# -*- coding: utf-8 -*-
"""
Created on Fri Mar 2 15:36:44 2018

@author: gg
"""

import xml.dom.minidom
import os

save_dir = 'E:\GCNCode'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f = open(os.path.join(save_dir, 'substance.txt'), 'w')

DOMTree = xml.dom.minidom.parse('E:\GCNCode\substances.xml')
annotation = DOMTree.documentElement

objects = annotation.getElementsByTagName("substance")


for object in objects:
    # 只需取smiles这一个标签的数据
    bbox = object.getElementsByTagName("smiles")[0]
    lefttopx = bbox.childNodes[0].data
    print(lefttopx)
    loc = lefttopx
    f.write(str(loc) + '\n')
f.close()

