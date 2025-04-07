import dataclasses
import numpy as np
from xml.etree import ElementTree

#TODO: Add citation for code

@dataclasses.dataclass
class Ink:
    strokes: list
    annotations: dict

def read_inkml_file(filename):
    with open(filename, "r") as f:
        root = ElementTree.fromstring(f.read())
    
    strokes, annotations = [], {}
    
    for element in root:
        tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')
        if tag_name == 'annotation':
            annotations[element.attrib.get('type')] = element.text

        elif tag_name == 'trace':
            points = element.text.split(',')
            stroke_x, stroke_y, stroke_t = [], [], []
            for point in points:
                x, y, t = point.split(' ')
                stroke_x.append(float(x))
                stroke_y.append(float(y))
                stroke_t.append(float(t))
            strokes.append(np.array((stroke_x, stroke_y, stroke_t)))

    return Ink(strokes=strokes, annotations=annotations)