from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from numpy import asarray
from numpy.random import randint
from scipy.interpolate import splprep, splev, UnivariateSpline
from matplotlib.patches import Ellipse

import numpy as np
import cv2



class TRACK_MASK():
    '''
    '''
    def __init__(self):
        '''
        '''
        # image config
        self.wm = 400       # width mask
        self.hm = 400       # height mask

    def frame_mask(self):
        '''
        '''
        px = 1/plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.wm*px,self.hm*px))
        plt.style.use('dark_background')
        plt.axis('off')

    def make_arrow(self,splx,sply):
        '''
        Make the arrow head at the end of the spline..
        '''
        arr_hd_len = 0.1
        dx = splx[-1] - splx[-2]
        dy = sply[-1] - sply[-2]
        arr_len = np.sqrt(dx**2 + dy**2)
        arr_dx = dx / arr_len * arr_hd_len
        arr_dy = dy / arr_len * arr_hd_len
        arr_start = (splx[-1] - arr_dx, sply[-1] - arr_dy)
        arr_end = (splx[-1], sply[-1])
        hw = 1.5*arr_hd_len
        hl = 3*arr_hd_len
        plt.arrow(*arr_start, dx, dy, head_width=hw, head_length=hl, color='white')

    def spline_3pts(self,x,y):
        '''
        Make a spline through 3 points..
        '''
        t = np.linspace(0, 1, 100)
        # Spline interpolation
        tck,u = splprep([x, y], k=2)
        splx, sply = splev(np.linspace(0, 1, 100), tck)

        return splx, sply

    def make_masks(self, invert_y=False, debug=[]):
        '''
        '''
        self.frame_mask()

        for i in range(self.nb_disks):
            x,y = [],[]
            for t in self.hist_centers:
                x += [t[i][0]]
                y += [t[i][1]]
            print(f'using x={x}')
            print(f'using y={y}')

            splx, sply = self.spline_3pts(x,y)
            plt.plot(splx, sply, 'w-')
            if 0 in debug:
                plt.plot(x, y, 'ro')
            self.make_arrow(splx,sply)

        # Display the plot
        plt.xlim(0,self.wm)
        plt.ylim(0,self.hm)
        #plt.axis('equal')
        plt.show()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig('track_masks.png')
        ###
        img_col = cv2.imread('composite.png')
        img_track = cv2.imread('track_masks.png')
        img_superp = cv2.addWeighted(img_col, 0.5, img_track, 0.5, 0)
        cv2.imwrite('superp.png', img_superp)

class SYNTH_DISKS(TRACK_MASK):
    '''
    '''
    def __init__(self, nb_disks=5):
        '''
        '''
        TRACK_MASK.__init__(self)
        # image config
        self.w = 400
        self.h = 400
        ###
        self.bckgd = (0, 0, 0)  #
        self.disk_col = (255, 255, 255)  #
        self.radius = 10  #
        self.nb_disks = nb_disks  #
        self.ldisks_centers = []
        self.hist_centers = []

    def black_image(self):
        '''
        '''
        self.compos_img = np.zeros((self.w,self.h,3))

    def create_image(self):
        '''
        '''
        # self.image = Image.new("RGB", (self.w, self.h), self.bckgd)
        # self.draw = ImageDraw.Draw(self.image)

        px = 1/plt.rcParams['figure.dpi']
        fig, self.ax = plt.subplots(figsize=(self.wm*px,self.hm*px),
                                    facecolor='black')
        #plt.figure()
        #plt.style.use('dark_background')
        self.ax.axis('off')


    def init_disks(self,debug=[1]):
        '''
        '''
        for i in range(self.nb_disks):
            x,y = randint(0,self.w), randint(0,self.h)
            self.ldisks_centers += [[x,y]]
        if 1 in debug:
            print(f'self.ldisks_centers = {self.ldisks_centers}')

    def move_disks(self):
        '''
        '''
        self.hist_centers += [self.ldisks_centers]
        new_list_centers = []
        for x,y in self.ldisks_centers:
            x0,y0 = x,y
            x += randint(-self.radius,self.radius)
            y += randint(-self.radius,self.radius)
            if x < 0 or x > self.w:
                x=x0
            if y < 0 or y > self.h:
                y=y0
            new_list_centers += [[x,y]]

        self.ldisks_centers = new_list_centers

    def add_ellipse(self,x,y):
        '''
        '''
        ellipse = Ellipse(xy=(x/self.w,y/self.h),
                          width=self.radius/self.w,
                          height=self.radius/self.h,
                          edgecolor='w', fc='w')
        self.ax.add_patch(ellipse)

    def draw_disks(self):
        '''
        '''
        for x,y in self.ldisks_centers:
            self.add_ellipse(x,y)

    def make_images(self, name_img, nb_imgs=1):
        '''
        '''
        self.init_disks()
        self.black_image()
        for i in range(nb_imgs):
            self.create_image()
            self.draw_disks()
            name_png = f'{name_img}{i}.png'
            plt.savefig(name_png)
            self.image = cv2.imread(name_png)
            self.compos_img[:,:,i] = asarray(self.image)[:,:,0]
            # self.save_image(name_img=f'{name_img}{i}.png')

            print(name_img)
            self.move_disks()
        cv2.imwrite('composite.png', self.compos_img)
        print('create the composite image..')
