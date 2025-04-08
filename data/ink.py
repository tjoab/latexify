import dataclasses
import numpy as np
from xml.etree import ElementTree
import cairo
import math
import PIL.Image

'''
This code is adapted from https://github.com/google-research/google-research/tree/master/mathwriting
and is accompanied with the MathWriting dataset in order to read and render inkML files.
'''

@dataclasses.dataclass
class Ink:
    """Represents a single ink, as read from an InkML file."""
    strokes: list
    annotations: dict


def read_inkml_file(filename: str) -> Ink:
    """Simple reader for MathWriting's InkML files."""
    with open(filename, "r") as f:
        root = ElementTree.fromstring(f.read())

    strokes = []
    annotations = {}

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


def cairo_to_pil(surface: cairo.ImageSurface) -> PIL.Image.Image:
  """Converts a ARGB Cairo surface into an RGB PIL image."""
  size = (surface.get_width(), surface.get_height())
  stride = surface.get_stride()
  
  with surface.get_data() as memory:
    return PIL.Image.frombuffer(
        'RGB', size, memory.tobytes(), 'raw', 'BGRX', stride
    )


def render_ink(
    ink: Ink,
    *,
    margin: int = 10,
    stroke_width: float = 1.5,
    stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PIL.Image.Image:
  """Renders an ink as a PIL image using Cairo. The image size is chosen to fit the 
  entire ink while having one pixel per InkML unit.

  Parameters
  ----------
  margin : int, optional (default is 10)
    size of the blank margin around the image (pixels)
  stroke_width : float, optional (default is 1.5)
    width of each stroke (pixels)
  stroke_color : tuple[float, float, float], optional (default is (0.0, 0.0, 0.0))
    color to paint the strokes with
  background_color : tuple[float, float, float], optional (default is (1.0, 1.0, 1.0))
    color to fill the background with

  Returns
  -------
  PIL.Image.Image
    Rendered ink, as a PIL image
  """

  # Compute transformation to fit the ink in the image.
  xmin, ymin = np.vstack([stroke[:2].min(axis=1) for stroke in ink.strokes]).min(axis=0)
  xmax, ymax = np.vstack([stroke[:2].max(axis=1) for stroke in ink.strokes]).max(axis=0)
  width = int(xmax - xmin + 2*margin)
  height = int(ymax - ymin + 2*margin)

  shift_x = - xmin + margin
  shift_y = - ymin + margin

  def apply_transform(ink_x: float, ink_y: float):
    return ink_x + shift_x, ink_y + shift_y

  # Create the canvas with the background color
  surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
  ctx = cairo.Context(surface)
  ctx.set_source_rgb(*background_color)
  ctx.paint()

  # Set pen parameters
  ctx.set_source_rgb(*stroke_color)
  ctx.set_line_width(stroke_width)
  ctx.set_line_cap(cairo.LineCap.ROUND)
  ctx.set_line_join(cairo.LineJoin.ROUND)

  for stroke in ink.strokes:
    if len(stroke[0]) == 1:
      # For isolated points we just draw a filled disk with a diameter equal
      # to the line width.
      x, y = apply_transform(stroke[0, 0], stroke[1, 0])
      ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
      ctx.fill()

    else:
      ctx.move_to(*apply_transform(stroke[0,0], stroke[1,0]))

      for ink_x, ink_y in stroke[:2, 1:].T:
        ctx.line_to(*apply_transform(ink_x, ink_y))
      ctx.stroke()

  return cairo_to_pil(surface)



'''
Example Usage
-------------
import os

path_to_inkml = os.path.join('./data/mathwriting-2024-excerpt', 'train', '000aa4c444cba3f2.inkml')

ink = read_inkml_file(path_to_inkml)
render_ink(ink)
'''


