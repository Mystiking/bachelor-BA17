import math
import cairo
 
class Draw(object):
    def __init__(self, w, h):
        self.WIDTH = w
        self.HEIGHT = h
        self.surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(self.WIDTH, self.HEIGHT)
        self.ctx.set_line_width(0.01)
        self.ctx.set_source_rgb(0, 0, 0)
 
    def draw_grid(self, n):
        s = 1./n
        for i in range(n):
            for j in range(n):
                self.ctx.rectangle(i*s, j*s, s, s)
        self.ctx.stroke()
 
    def draw_right_arrow(self, i, j, n):
        s = 1./n
        length = 2*s/5
        # Main line
        self.ctx.move_to(i*s + s/2, j*s + s/2)
        self.ctx.line_to(i*s + s/2 + length, j*s + s/2)
        # "Upper" line
        self.ctx.move_to(i*s + s/2 + length, j*s + s/2)
        self.ctx.line_to(i*s + s/2 + length / 3, j*s + s/2 - length / 3)
        # "Lower" line
        self.ctx.move_to(i*s + s/2 + length, j*s + s/2)
        self.ctx.line_to(i*s + s/2 + length / 3, j*s + s/2 + length / 2)
        self.ctx.stroke()
 
    def draw_left_arrow(self, i, j, n):
        s = 1./n
        length = 2*s/5
        # Main line
        self.ctx.move_to(i*s + s/2, j*s + s/2)
        self.ctx.line_to(i*s + s/2 - length, j*s + s/2)
        # "Upper" line
        self.ctx.move_to(i*s + s/2 - length, j*s + s/2)
        self.ctx.line_to(i*s + s/2 - length / 3, j*s + s/2 - length / 2)
        # "Lower" line
        self.ctx.move_to(i*s + s/2 - length, j*s + s/2)
        self.ctx.line_to(i*s + s/2 - length / 3, j*s + s/2 + length / 2)
        self.ctx.stroke()
 
    def draw_up_arrow(self, i, j, n):
        s = 1./n
        length = 2*s/5
        # Main line
        self.ctx.move_to(i*s + s/2, j*s + s/2)
        self.ctx.line_to(i*s + s/2, j*s + s/2 - length)
        # "Right" line
        self.ctx.move_to(i*s + s/2, j*s + s/2 - length)
        self.ctx.line_to(i*s + s/2 + length / 3, j*s + s/2 - (length / 3))
        # "left" line
        self.ctx.move_to(i*s + s/2, j*s + s/2 - length)
        self.ctx.line_to(i*s + s/2 - length / 3, j*s + s/2 - length / 3)
        self.ctx.stroke()
 
    def draw_down_arrow(self, i, j, n):
        s = 1./n
        length = 2*s/5
        # Main line
        self.ctx.move_to(i*s + s/2, j*s + s/2)
        self.ctx.line_to(i*s + s/2, j*s + s/2 + length)
        # "Right" line
        self.ctx.move_to(i*s + s/2, j*s + s/2 + length)
        self.ctx.line_to(i*s + s/2 - length / 3, j*s + s/2 + (length / 3))
        # "left" line
        self.ctx.move_to(i*s + s/2, j*s + s/2 + length)
        self.ctx.line_to(i*s + s/2 + length / 3, j*s + s/2 + (length / 3))
        self.ctx.stroke()
 
    def write(self, name):
        self.surface.write_to_png(name)
