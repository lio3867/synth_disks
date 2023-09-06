from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from numpy import asarray
from numpy.random import randint
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.linalg import norm
from matplotlib.patches import Ellipse
import pickle as pkl

import os
import numpy as np
import cv2



class TRACK_MASK():
    '''
    '''
    def __init__(self):
        '''
        '''
        # image config
        self.wm = 512       # width of the image for mask
        self.hm = 512       # height of the image for mask

    def frame_mask(self):
        '''
        '''
        px = 1/plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.wm*px,self.hm*px))
        plt.style.use('dark_background')
        plt.axis('off')

    def make_arrow(self, splx, sply, arrsize=3, debug=[]):
        '''
        Make the arrow head at the end of the spline..
        '''
        arr_hd_len = 0.1
        dx = splx[-1] - splx[-2]
        dy = sply[-1] - sply[-2]
        arr_len = np.sqrt(dx**2 + dy**2)
        arr_dx = dx / arr_len * arr_hd_len
        arr_dy = dy / arr_len * arr_hd_len
        if 1 in debug:
            print(f'arr_dx = {arr_dx}')
            print(f'arr_dy = {arr_dy}')
        arr_start = (splx[-1] - arr_dx, sply[-1] - arr_dy)
        arr_end = (splx[-1], sply[-1])
        hw = arrsize*arr_hd_len                 # head width
        hl = 2*arrsize*arr_hd_len               # head length
        plt.arrow(*arr_start, dx, dy, head_width=hw, head_length=hl, color='white')

    def spline_3pts(self, x, y):
        '''
        Make a spline through 3 points..
        '''
        t = np.linspace(0, 1, 100)
        # Spline interpolation
        tck,u = splprep([x, y], k=2)
        splx, sply = splev(np.linspace(0, 1, 100), tck)

        return splx, sply

    def make_masks(self, ind, invert_y=False, arrsize=6, debug=[]):
        '''
        Make the trajectory masks..
        arrsize: size of the arrow
        '''
        self.frame_mask()

        for i in range(self.nb_objs):
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
            self.make_arrow(splx,sply,arrsize=arrsize)

        # Display the plot
        plt.xlim(0,self.wm)
        plt.ylim(0,self.hm)
        #plt.axis('equal')
        plt.show()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig('results/track_masks.png')
        try:
            plt.savefig(f'results/masks/{ind}.png')
        except:
            print('Cannot save the masks')
        ###
        img_col = cv2.imread('results/trace_img.png')
        img_track = cv2.imread('results/track_masks.png')
        img_superp = cv2.addWeighted(img_col, 0.5, img_track, 0.5, 0)
        cv2.imwrite('results/superp.png', img_superp)

class GEN_MOV_OBJ(TRACK_MASK):
    '''
    '''
    def __init__(self, nb_objs=5):
        '''
        '''
        TRACK_MASK.__init__(self)
        # image config
        self.w = 512   # width
        self.h = 512   # height
        ###
        self.bckgd = (0, 0, 0)  #
        self.disk_col = (255, 255, 255)  #
        self.radius = 10  #
        self.edge = 10
        self.nb_objs = nb_objs  #
        self.list_type_obj = []
        self.in_cone = np.pi/6

        self.images_and_masks_folders()

    def images_and_masks_folders(self):
        '''
        Prepare the folders for the training set..
        '''
        try:
            os.mkdir('results/images')
        except:
            print('images folder yet exists')
        try:
            os.mkdir('results/masks')
        except:
            print('masks folder yet exists')

    def black_image(self, nblay=3):
        '''
        Composite image
        '''
        self.compos_img = np.zeros((self.w,self.h,nblay))
        self.trace_img = np.zeros((self.w,self.h,3))

    def create_image(self):
        '''
        '''
        px = 1/plt.rcParams['figure.dpi']
        self.fig, self.ax = plt.subplots(figsize=(self.wm*px,self.hm*px),
                                         facecolor='black')
        self.ax.axis('off')

    def init_objects(self,debug=[1]):
        '''
        '''
        for i in range(self.nb_objs):
            x,y = randint(0,self.w), randint(0,self.h)
            self.lobjs_centers += [[x,y]]
        if 1 in debug:
            print(f'self.lobjs_centers = {self.lobjs_centers}')

    def move_objects(self):
        '''
        Move randomly the disks
        next move in the angle in_cone
        '''
        self.hist_centers += [self.lobjs_centers]
        new_list_centers = []
        for i,(x,y) in enumerate(self.lobjs_centers):
            x0,y0 = x,y
            if len(self.hist_centers) > 1 and self.in_cone:
                dl = randint(1,self.radius)
                x00,y00 = self.hist_centers[-1][i]
                x11,y11 = self.hist_centers[-2][i]
                vec = np.array([x11-x00, y11-y00])
                vec_norm = vec/norm(vec)
                theta = np.arctan(vec_norm[1]/vec_norm[0])
                rand_ang = np.random.uniform(-self.in_cone, self.in_cone)
                new_ang = theta + rand_ang
                x += dl*np.cos(new_ang)
                y += dl*np.sin(new_ang)
            else:
                x += randint(-self.radius,self.radius)
                y += randint(-self.radius,self.radius)
            # stay in the limits
            if x < 0 or x > self.w:
                x=x0
            if y < 0 or y > self.h:
                y=y0
            new_list_centers += [[x,y]]

        self.lobjs_centers = new_list_centers

    def add_ellipse(self,x,y):
        '''
        '''
        ellipse = Ellipse(xy=(x/self.w,y/self.h),
                          width=self.radius/self.w,
                          height=self.radius/self.h,
                          edgecolor='w', fc='w')
        self.ax.add_patch(ellipse)

    def add_square(self,x,y):
        '''
        '''
        rect = patches.Rectangle((self.edge, self.edge), x,y,
                                  linewidth=1,
                                  edgecolor='r',
                                  facecolor='none')
        self.ax.add_patch(rect)

    def draw_shapes(self):
        '''
        '''
        for x,y in self.lobjs_centers:
            self.add_ellipse(x,y)

    def make_images(self, ind, name_img, nb_time_pts=1):
        '''
        ind : index of the image.. eg:img0.png etc..
        '''
        self.lobjs_centers = []
        self.hist_centers = []
        self.init_objects()
        self.black_image(nblay=nb_time_pts)
        for i in range(nb_time_pts):
            print(f'time point num {i}')
            self.create_image()
            self.draw_shapes()
            name_png = f'results/{name_img}{i}.png'
            plt.savefig(name_png, facecolor=self.fig.get_facecolor())
            plt.close()
            self.image = cv2.imread(name_png)
            self.compos_img[:,:,i] = asarray(self.image)[:,:,0]
            self.trace_img[:,:,i%3] += asarray(self.image)[:,:,0]
            # self.save_image(name_img=f'{name_img}{i}.png')
            print(name_img)
            self.move_objects()
        cv2.imwrite('results/trace_img.png', self.trace_img)
        with open('composite.pk', 'wb') as f:
            pkl.dump(self.compos_img, f)
        try:
            cv2.imwrite(f'results/images/{ind}.png', self.compos_img)
        except:
            print('Cannot save the images')
        print('create the composite image..')
