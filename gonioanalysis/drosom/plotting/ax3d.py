'''Alternative 3D plotting using hosguibase's render3d
'''

import os
import math
import tkinter as tk

import numpy as np
from panda3d.core import (
        NodePath,
        DirectionalLight,
        AmbientLight,
        Filename,
        PNMImage,
        Texture,
        TextNode,
        )
from direct.gui.DirectGui import DirectFrame, DirectLabel

import devjoni.guibase as gb
from devjoni.hosguibase.render3d import SceneWidget, SceneObject

from gonioanalysis.directories import CODE_ROOTDIR
from gonioanalysis.coordinates import rotate_along_arbitrary


named_colors = {
        'white': (1,1,1,1),
        'black': (0,0,0,1),
        'red': (1,0,0,1),
        'blue': (0,0,1,1),
        'green': (0,1,0,1),
        'orange': (1,0.5,0,1),
        'pink': (1,0.75,0.75,1),
        'gray': (0.5,0.5,0.5,1),
        'magenta': (1,0,1,1),
        }


class WrapTk():
    def __init__(self, tkwidget):
        self.tk = tkwidget


class RotateButtons(gb.FrameWidget):
    '''Rotate buttons
    '''
    def __init__(self, parent, obj, use_buttons=False):
        super().__init__(parent)
        self.parent = parent
        self.obj = obj

        if use_buttons:
            self.horp = gb.ButtonWidget(self, 'h+',
                                        command=self.rhorp)
            self.horp.grid(row=0,column=0)
            self.horm = gb.ButtonWidget(self, 'h-',
                                        command=self.rhorm)
            self.horm.grid(row=0,column=1)

            self.verp = gb.ButtonWidget(self, 'v+',
                                        command=self.rverp)
            self.verp.grid(row=0,column=2)
            self.verm = gb.ButtonWidget(self, 'v-',
                                        command=self.rverm)
            self.verm.grid(row=0,column=3)
        else:
            self.hor_slider = gb.SliderWidget(
                    self, from_=-90, to=90)
            self.hor_slider.grid(row=0, column=0, sticky='WE',
                                 row_weight=0)
            self.hor_slider.set_command(self.set_hor)

            self.ver_slider = gb.SliderWidget(
                    self, from_=-90, to=90)
            self.ver_slider.grid(row=1, column=0, sticky='WE',
                                 row_weight=0)
            self.ver_slider.set_command(self.set_ver)


    def set_ver(self, value):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(rot[0], float(value), rot[2])
        self.parent.update()

    def set_hor(self, value):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(float(value)+180, rot[1], rot[2])
        self.parent.update()

    def rhorp(self):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(rot[0]+10, rot[1], rot[2])
        self.parent.update()
    
    def rhorm(self):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(rot[0]-10, rot[1], rot[2]) 
        self.parent.update()

    def rverp(self):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(rot[0], rot[1]+10, rot[2])
        self.parent.update()

    def rverm(self):
        rot = self.obj.get_hpr()
        self.obj.set_hpr(rot[0], rot[1]-10, rot[2])
        self.parent.update()


class Cp3d(gb.FrameWidget):
    '''Use in place of CanvasPlotter
    '''

    def __init__(self, parent):
        if isinstance(parent, tk.Frame):
            parent = WrapTk(parent)
        super().__init__(parent)

        self.ax3d = Ax3d(self)
        self.ax3d.grid(row=0, column=0)
        
        self.obj = SceneObject('central')
        self.obj.scene = self.ax3d.scene

        self.ax3d.nodeparent = self.obj
        self.ax3d.nodeparent.set_hpr(180,0,0)

        self.buttons = RotateButtons(self, self.obj)
        self.buttons.grid(row=1, column=0)

    def get_figax(self):
        return self.ax3d, self.ax3d

    def update(self):
        self.ax3d.scene.render()

class Ax3d(gb.FrameWidget):
    '''Use in place of matplotlib ax.
    '''

    def __init__(self, parent, nodeparent=None):
        super().__init__(parent)

        self.nodeparent = nodeparent

        self.scene = SceneWidget(self)
        self.scene.grid()

        self.scene.camera.set_pos(0,-9,0)
        self.scene.camera.set_hpr(0,0,0)
        

        self._arrow = SceneObject('arrow')
        self._arrow.scene = self.scene
        self._arrow.load_model(
                Filename.fromOsSpecific(
                    os.path.join(CODE_ROOTDIR, 'progdata', 'arrow.egg')
                    )
                )
        self._arrow.np.set_scale(0.05)
        self._arrow.np.setTwoSided(True)
        self._arrow.np.set_hpr(180,-90,0)
        self._arrow.np.flatten_strong()
        self._arrow.np.detachNode()

        self.arrows = []
       
    

        # Head
        self.head = SceneObject('head')
        self.head.scene = self.scene
        self.head.load_model(
                Filename.fromOsSpecific(
                    os.path.join(CODE_ROOTDIR, 'progdata', 'head.egg')
                    )
                )
        self.head.np.detachNode()
        self.head.np.setScale(3.3)
        self.head.np.setColorScale((0.8,0.8,0.8,1))
        #self.head.np.setShaderAuto()
        self._enable_lights()

        self.scene.showbase.setBackgroundColor((1,1,1,1))


        self.bar = None


    def show_head(self):
        self.head.np.reparentTo(self.nodeparent.np)

    def hide_head(self):
        self.head.np.detachNode()

    def _enable_lights(self):
        render = self.scene._rendernp
        #render.setShaderAuto()

        ambient = AmbientLight('ambient')
        ambient.setColor((0.3, 0.3, 0.3, 1))
        ambient_np = render.attachNewNode(ambient)
        render.setLight(ambient_np)
        
        for p in [0,1]:
            pitch = -30+p*60
            for heading in [0+p*60,120+p*60,240+p*60]:
                lamp = DirectionalLight(f'lamp-h{heading}-p{p}')
                lamp.setColor((0.4+p*0.2, 0.4+p*0.2, 0.4+p*0.2, 1))
                lamp_np = render.attachNewNode(lamp)
                lamp_np.setHpr(heading, pitch, 0)
                render.setLight(lamp_np)

    def set_axis_off(self):
        pass

    def set_xlabel(self, label):
        pass

    def set_ylabel(self, label):
        pass

    def set_zlabel(self, label):
        pass

    def add_arrow(self, x0,y0,z0,x1,y1,z1,
                  color=None, mutation_scale=6):

        arrow = NodePath('arrow')
        self._arrow.np.instance_to(arrow)

        arrow.reparent_to(self.nodeparent.np)

        h = math.degrees(math.atan2(y0,x0))+90

        dxy0 = math.sqrt(x0**2+y0**2)
        p = math.degrees(math.atan2(z0,dxy0))

        # Calculate roll by rotating the arrow to (0,1,0)
        p0 = np.array([(x0,y0,z0)])
        p1 = np.array([(x1,y1,z1)])
        
        p0 = rotate_along_arbitrary((0,0,1), p0, math.radians(h))[0]
        p1 = rotate_along_arbitrary((1,0,0), p1, math.radians(p))[0]
        
        pdx = p1[0]-p0[0]
        pdz = p1[2]-p0[2]
        r = math.degrees(math.atan2(pdx,pdz))

        arrow.set_hpr(h,-(p),-r+180) 

        if x0 < 0:
            shift = -0.6
        else:
            shift = 0.6 

        nx0 = x0*0.8+shift
        ny0 = y0*0.7
        nz0 = z0*0.8
     
        arrow.set_pos(nx0, ny0, nz0)
        if isinstance(color, str):
            colscale = named_colors.get(color, (0.9,0.9,0.9))
            arrow.setColor(colscale)  
        else:
            arrow.setColor(color[0], color[1], color[2])  

        self.arrows.append(arrow)

    def _create_colorbar(self, colors, values):
        Nc = len(colors)
        Nv = len(values)
        if Nc != Nv:
            raise ValueError(
                    f'colors and values different lengths ({Nc}, {Nv})')

        bar = PNMImage(x_size=1, y_size=Nc, num_channels=3, maxval=255)
        for i, col in enumerate(colors):
            bar.setXel(0,i,col[0], col[1], col[2])
        return bar


    def set_colorbar(self, colors, values, labels, vertical=True):
        self.clear_colorbar()

        bar = DirectFrame(parent=self.scene.showbase.aspect2d,
                          scale=(0.025,1,0.5),
                          pos=(0.7,0,0))
        bar.setR(180)
        
        N = len(labels)
        for i, label in enumerate(labels):
            text = DirectLabel(
                    parent=bar, text=label,
                    scale=(2,1,0.11), frameColor=(0,0,0,0),
                    text_align=TextNode.ARight)
            text.set_transparency(True)
            text.setR(180)
            text.set_pos(1.5,0,-1 + i*(4/N) + 0.025)

        barim = self._create_colorbar(colors, values)
        bartex = Texture()
        bartex.load(barim)
        bartex.setWrapU(Texture.WM_clamp)
        bartex.setWrapV(Texture.WM_clamp)

        bar['image'] = bartex
        self.bar = bar

    def clear_colorbar(self):
        if self.bar is not None:
            self.bar.destroy()
            self.bar = None
        

    def plot_surface(self, X, Y, Z, color=None):
        pass

    def plot(self, X, Y, Z):
        pass

    def clear(self):
        
        for arrow in self.arrows:
            arrow.removeNode()

        self.arrows = []
        
        self.hide_head()

    def view_init(self, elev, azim):
        '''
        elev, azim : float
            Rotations in degrees
        '''
        self.nodeparent.set_hpr(-azim-90,-elev,0)
        self.show_head()

    @property
    def azim(self):
        return -self.nodeparent.get_hpr()[0]-90

    @azim.setter
    def azim(self, value):
        h,p,r = self.nodeparent.get_hpr()
        self.nodeparent.set_hpr(-value-90, p, r)
    
    @property
    def elev(self):
        return -self.nodeparent.get_hpr()[1]

    @elev.setter
    def elev(self, value):
        h,p,r = self.nodeparent.get_hpr()
        return self.nodeparent.set_hpr(h, value, r)

