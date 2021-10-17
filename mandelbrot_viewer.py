# ref_1: https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
# ref_2: https://stackoverflow.com/questions/16500656

import taichi as ti
from numpy import linspace, array
from scipy.interpolate import pchip_interpolate

ti.init(arch=ti.cpu)

# canvas size
width = 640
height = 360

# initial config
center_x = -0.5
center_y = 0
zoom = 150

# misc
zoom_rate = 1.2
max_iter = 500
colormap_size = 1000
colormap = pchip_interpolate(
    [0, 0.16, 0.42, 0.6425, 0.8575, 1],
    array([[0, 7, 100], [32, 107, 203], [237, 255, 255], [255, 170, 0], [0, 2, 0], [0, 7, 100]]) / 255,
    linspace(0, 1, colormap_size)
    ).flatten()

pixels = ti.Vector.field(3, dtype=float, shape=(width, height))
gui = ti.GUI("Mandelbrot Viewer", res=(width, height))

@ti.func
def iteration(x, y):
    c = ti.Vector([x, y])
    z = c
    count = 1.0
    while z.norm() <= 2 and count < max_iter:
        # z = z^2 + c
        z = ti.Vector([z[0]**2 - z[1]**2, z[0] * z[1] * 2]) + c
        count += 1.0
    if count < max_iter:
        # smooth color
        count += 1.0 - ti.log(ti.log(ti.cast(z.norm(), ti.f32)) / ti.log(2)) / ti.log(2)
    return count

@ti.kernel
def paint(center_x: ti.f64, center_y: ti.f64, zoom: ti.f64, colormap: ti.ext_arr()):
    for i, j in pixels:
        x = center_x + (i - width / 2 + 0.5) / zoom
        y = center_y + (j - height / 2 + 0.5) / zoom
        index = int(iteration(x, y) / max_iter * colormap_size)
        for k in ti.static(range(3)):
            pixels[i, j][k] = colormap[3 * index + k]

# GUI
gui.fps_limit = 10
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == ti.GUI.LMB:
            # left click: record position
            mouse_x0, mouse_y0 = gui.get_cursor_pos()
            center_x0, center_y0 = center_x, center_y
        elif e.key == ti.GUI.WHEEL:
            # scroll: zoom
            mouse_x, mouse_y = gui.get_cursor_pos()
            if e.delta[1] > 0:
                zoom_new = zoom * zoom_rate
            elif e.delta[1] < 0:
                zoom_new = zoom / zoom_rate
            center_x += (mouse_x - 0.5) * width * (1 / zoom - 1 / zoom_new)
            center_y += (mouse_y - 0.5) * height * (1 / zoom - 1 / zoom_new)
            zoom = zoom_new
        elif e.key == ti.GUI.SPACE:
            # space: print info
            print(f'center_x={center_x}, center_y={center_y}, zoom={zoom}')
    if gui.is_pressed(ti.GUI.LMB):
        # drag: move
        mouse_x, mouse_y = gui.get_cursor_pos()
        center_x = center_x0 + (mouse_x0 - mouse_x) * width / zoom
        center_y = center_y0 + (mouse_y0 - mouse_y) * height / zoom

    paint(center_x, center_y, zoom, colormap)
    gui.set_image(pixels)
    gui.show()
