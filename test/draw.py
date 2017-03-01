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
        self.ctx.set_source_rgb(255, 255, 255)
        self.ctx.rectangle(0, 0, 1, 1)
        self.ctx.fill()
        self.ctx.set_source_rgb(0, 0, 0)

    def draw_grid(self, n, m):
        s = 1./(max(n, m))
        for i in range(m):
            for j in range(n):
                self.ctx.rectangle(i*s, j*s, s, s)
        self.ctx.stroke()

    def fill_rectangle(self, j, i, n, m, r, g, b):
        s = 1./max(n, m)
        length = 2*s/5
        self.ctx.set_source_rgb(r, g, b)
        self.ctx.rectangle(i*s, j*s, s, s)
        self.ctx.fill()
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.stroke()

    def draw_right_arrow(self, j, i, n, m):
        s = 1./max(n, m)
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
 
    def draw_left_arrow(self, j, i, n, m):
        s = 1./max(n, m)
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
 
    def draw_up_arrow(self, j, i, n, m):
        s = 1./max(n, m)
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
 
    def draw_down_arrow(self, j, i, n, m):
        s = 1./max(n, m)
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
