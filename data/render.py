import cairo
import math
import numpy as np
from PIL import Image

#TODO: Add citation for code
#TODO: need system level reqs for cairo to work be sure to include those in README... or dockerize?

def cairo_to_pil(surface: cairo.ImageSurface) -> Image.Image:
    size = (surface.get_width(), surface.get_height())
    stride = surface.get_stride()
    with surface.get_data() as memory:
        return Image.frombuffer('RGB', size, memory.tobytes(), 'raw', 'BGRX', stride)

def render_ink(ink, margin=10, stroke_width=1.5, stroke_color=(0.0, 0.0, 0.0), background_color=(1.0, 1.0, 1.0)):
    xmin, ymin = np.vstack([s[:2].min(axis=1) for s in ink.strokes]).min(axis=0)
    xmax, ymax = np.vstack([s[:2].max(axis=1) for s in ink.strokes]).max(axis=0)
    width = int(xmax - xmin + 2 * margin)
    height = int(ymax - ymin + 2 * margin)

    shift_x = -xmin + margin
    shift_y = -ymin + margin

    def apply_transform(x, y): return x + shift_x, y + shift_y

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*background_color)
    ctx.paint()

    ctx.set_source_rgb(*stroke_color)
    ctx.set_line_width(stroke_width)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)

    for stroke in ink.strokes:
        if len(stroke[0]) == 1:
            x, y = apply_transform(stroke[0, 0], stroke[1, 0])
            ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
            ctx.fill()
        else:
            ctx.move_to(*apply_transform(stroke[0, 0], stroke[1, 0]))
            for x, y in stroke[:2, 1:].T:
                ctx.line_to(*apply_transform(x, y))
            ctx.stroke()

    return cairo_to_pil(surface)
