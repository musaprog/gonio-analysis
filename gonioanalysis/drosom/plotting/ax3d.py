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
        )
import devjoni.guibase as gb
from devjoni.hosguibase.render3d import SceneWidget, SceneObject

from gonioanalysis.directories import CODE_ROOTDIR
from gonioanalysis.coordinates import rotate_along_arbitrary


named_colors = {
        'white': (1,1,1,1),
        'black': (0,0,0,1),
        'red': (1,0,1,1),
        'blue': (0,0,1,1),
        'green': (0,1,0,1),
        'orange': (1,0.5,0,1),
        'pink': (1,0.75,0.75,1),
        'gray': (0.5,0.5,0.5,1),
        }


class WrapTk():
    def __init__(self, tkwidget):
        self.tk = tkwidget


class RotateButtons(gb.FrameWidget):
    '''Rotate buttons
    '''
    def __init__(self, parent, obj):
        super().__init__(parent)
        self.parent = parent
        self.obj = obj

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

        #self.obj.load_model(
        #        os.path.join(CODE_ROOTDIR, 'progdata', 'monkey.egg')
        #        )


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
                os.path.join(CODE_ROOTDIR, 'progdata', 'arrow.egg')
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
                os.path.join(CODE_ROOTDIR, 'progdata', 'head.egg')
                )
        self.head.np.detachNode()
        self.head.np.setScale(3)
        self.head.np.setColorScale((0.8,0.8,0.8,1))
        self.head.np.setShaderAuto()

        self._enable_lights()

        self.scene.showbase.setBackgroundColor((1,1,1,1))

    def show_head(self):
        self.head.np.reparentTo(self.nodeparent.np)

    def hide_head(self):
        self.head.np.detachNode()

    def _enable_lights(self):
        render = self.scene._rendernp
        render.setShaderAuto()

        ambient = AmbientLight('ambient')
        ambient.setColor((0.3, 0.3, 0.3, 1))
        ambient_np = render.attachNewNode(ambient)
        render.setLight(ambient_np)
       
        for p in [0,1]:
            pitch = -30+p*60
            for heading in [0+p*60,120+p*60,240+p*60]:
                lamp = DirectionalLight(f'lamp-h{heading}')
                lamp.setColor((0.6, 0.6, 0.6, 1))
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

        dx = x1-x0
        dy = y1-y0
        dz = z1-z0

        dxy = math.sqrt(dx**2+dy**2)
        dxyz = math.sqrt(dx**2+dy**2+dz**2)

        h = math.degrees(math.atan2(dy,dx))
        p = math.degrees(math.atan2(dxy, dz))


        # Calculate roll by rotating the arrow to (0,1,0)
        p0 = np.array([(x0,y0,z0)])
        p1 = np.array([(x1,y1,z1)])
        
        p0 = rotate_along_arbitrary((0,0,1), p0, math.radians(h))[0]
        p1 = rotate_along_arbitrary((1,0,0), p1, math.radians(p))[0]
        
        pdx = p1[0]-p0[0]
        pdz = p1[2]-p0[2]
        r = math.degrees(math.atan2(pdx,pdz))

        arrow.set_hpr(h,-(p-90),r)

        #arrow.flatten_strong()
        #arrow.set_r(r)
        #arrow.flatten_strong()
        #arrow.set_p(-(p-90))
        #arrow.set_h(h)
        
        if x0 < 0:
            shift = -0.5
        else:
            shift = 0.5
        arrow.set_pos(x0+shift,y0*0.7,z0*0.8)
        
        colscale = named_colors.get(color, (0.9,0.9,0.9))
        arrow.setColor(colscale)  

        self.arrows.append(arrow)

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

    @property
    def elev(self):
        return -self.nodeparent.get_hpr()[1]
