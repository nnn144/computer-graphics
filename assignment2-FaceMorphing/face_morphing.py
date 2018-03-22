# import packages
import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.spatial import Delaunay
from os import listdir

"""
    Data type:
        path_id
        point_type
        x_rel
        y_rel
        point_id
        connectsFrom
        connectsTo
    Return:
        path_id
        point_type
        pointSet = (x_rel, y_rel)
        point_id
        connectsFrom
        connectsTo
"""
# read data from file
def get_data(file_name):
    with open(file_name) as f:
        # fist line is the number of points
        line = f.readline()
        
        # ignore comments
        if not line.startswith('#'):
            num_points = int(line)
            #create several np.array to store the data
            path_id = np.zeros(num_points)
            point_type = np.zeros(num_points)
            x_rel = np.zeros(num_points)
            y_rel = np.zeros(num_points)
            pointSet = np.zeros((num_points, 2))
            point_id = np.zeros(num_points)
            connectsFrom = np.zeros(num_points)
            connectsTo = np.zeros(num_points)

            #read data
            for i in range(0, num_points):
                line = f.readline().split(' ')
                path_id[i] = int(line[0])
                point_type[i] = int(line[1])
                x_rel[i] = float(line[2]) * 640
                y_rel[i] = float(line[3]) * 480
                pointSet[i] = np.array([x_rel[i], y_rel[i]])
                point_id[i] = int(line[4])
                connectsFrom[i] = int(line[5])
                connectsTo[i] = int(line[6])

    return path_id, point_type, pointSet, point_id, connectsFrom, connectsTo

#=====================================================================
# read data from .asf file
def read_asf(file_name):
    #create several lists to store the data
    path_id = []
    point_type = []
    x_rel = []
    y_rel = []
    pointSet = []
    point_id = []
    connectsFrom = []
    connectsTo = []
    with open(file_name) as f:
        for line in f:
            input_line = line.strip()

            # ignore comments
            if not line.startswith('#'):
                # first line shows the number of points, 2 chars
                if len(input_line) > 0 and len(input_line) < 5:
                    num_points = int(input_line)
                    #print (num_points)

                #read data
                elif len(input_line) > 10:
                    str_split = input_line.split(' ')
                    path_id.append(int(str_split[0]))
                    point_type.append(int(str_split[1]))
                    x_rel.append(float(str_split[2]) * 640)
                    y_rel.append(float(str_split[3]) * 480)
                    pointSet.append(np.array([float(str_split[2]) * 640, float(str_split[3]) * 480]))
                    point_id.append(int(str_split[4]))
                    connectsFrom.append(int(str_split[5]))
                    connectsTo.append((str_split[6]))
    
    pointSet = np.asarray(pointSet)
    return path_id, point_type, pointSet, point_id, connectsFrom, connectsTo

#=====================================================================
"""
Slight difference in pixel ordering in OpenCV and Matplotlib.
OpenCV follows BGR order, while matplotlib likely follows RGB order.
So the image need to re-order to show correctly
"""
def reordering(image):
    b1, g1, r1 = cv2.split(image)
    # re-order
    reorder = cv2.merge([r1, g1, b1])
    
    return reorder

#=====================================================================
"""
Alpha blending, combine two images
imput: src_image ===> source image
       dst_image ===> destination image
       alpha     ===> rate
return: image = alpha * src_image + (1 - alpha) * dst_image
"""
def combining(srd_image, dst_image, alpha):
    # split into b, g, r three channels
    b1, g1, r1 = cv2.split(srd_image)
    b2, g2, r2 = cv2.split(dst_image)
    # combine
    beta = 0
    b_final = cv2.multiply(alpha, b1) + cv2.multiply(beta, b2)
    g_final = cv2.multiply(alpha, g1) + cv2.multiply(beta, g2)
    r_final = cv2.multiply(alpha, r1) + cv2.multiply(beta, r2)
    # merge
    final = cv2.merge([b_final, g_final, r_final])
    return final

#=====================================================================
"""
create a mask according to the given coordinates
input: polygon_points ===> coordinates
       image          ===> original image
output: mask with the same size of the image
"""
def mask(polygon_points, image):
    #image = image.reshape(-1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    #roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillConvexPoly(mask, polygon_points.astype(dtype='int32'), ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#=====================================================================
"""
create a mask_inverse according to the given coordinates
input: polygon_points ===> coordinates
       image          ===> original image
output: mask with the same size of the image
"""
def mask_img(polygon_points, image):
    #image = image.reshape(-1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    #roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillConvexPoly(mask, polygon_points.astype(dtype='int32'), ignore_mask_color)
    
    return mask

#=====================================================================
"""
Transparency: image * alpha, alpha = [0, 1]
input: image, alpha
outut: image * alpha
"""
def img_mul(image, alpha):
    # split the image
    b, g, r = cv2.split(image)
    # multiplication
    b_final = cv2.multiply(alpha, b)
    g_final = cv2.multiply(alpha, g)
    r_final = cv2.multiply(alpha, r)
    # merge
    final = cv2.merge([b_final, g_final, r_final])
    
    return final

#=====================================================================
"""
Face morphing
input: src, dst: source and destination images
     mid_img: image in between, the actual output
     tri_src, tri_dst, tri_mid: delaunay triangles
     alpha: mid_img = alpha * src + (1 - alpha) * dst
"""
def morphing(src, dst, mid_img, tri_src, tri_dst, tri_mid, alpha):
    # Find bounding rectangle points for each triangle
    # to prevent gaps
    # cv2.boundingRect return [x, y, w, h]
    # (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    rec_src = cv2.boundingRect(tri_src.astype(dtype='float32'))
    rec_dst = cv2.boundingRect(tri_dst.astype(dtype='float32'))
    rec_mid = cv2.boundingRect(tri_mid.astype(dtype='float32'))
    # used for cropping the mask
    src_tri_m = []
    dst_tri_m = []
    mid_tri_m = []
    for i in range(0, 3):
        src_tri_m.append(((tri_src[i, 0] - rec_src[0]),(tri_src[i, 1] - rec_src[1])))
        dst_tri_m.append(((tri_dst[i, 0] - rec_dst[0]),(tri_dst[i, 1] - rec_dst[1])))
        mid_tri_m.append(((tri_mid[i, 0] - rec_mid[0]),(tri_mid[i, 1] - rec_mid[1])))
    # mask of the rectangle
    #mask = mask_img(tri_src, np.zeros([rec_mid_pts[3], rec_mid_pts[2], 3]))
    mask = np.zeros((rec_mid[3], rec_mid[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(mid_tri_m), (1.0, 1.0, 1.0), 16, 0);
    
    # warping
    # src / dst to mid
    src_bound_rect = src[rec_src[1]:rec_src[1] + rec_src[3], rec_src[0]:rec_src[0] + rec_src[2]]
    dst_bound_rect = dst[rec_dst[1]:rec_dst[1] + rec_dst[3], rec_dst[0]:rec_dst[0] + rec_dst[2]]
    # warp size = [rec_mid_w, rec_mid_h], because from src/dst to mid
    warp_size = (rec_mid[2], rec_mid[3])
    # warping matrix
    src_warp_mat = cv2.getAffineTransform(np.float32(src_tri_m), np.float32(mid_tri_m))
    dst_warp_mat = cv2.getAffineTransform(np.float32(dst_tri_m), np.float32(mid_tri_m))

    # warping image
    warp_src = cv2.warpAffine(src_bound_rect, src_warp_mat, (warp_size[0], warp_size[1]))
    warp_dst = cv2.warpAffine(dst_bound_rect, dst_warp_mat, (warp_size[0], warp_size[1]))
    # alpha blending
    mid_img_rect = combining(warp_src, warp_dst, 1 / 37)
    #mid_img_rect = img_mul(warp_src, 1 / 37)
    # save the mask to the mid image
    mid_img[rec_mid[1]:rec_mid[1] + rec_mid[3], rec_mid[0]:rec_mid[0] + rec_mid[2]] = \
    mid_img[rec_mid[1]:rec_mid[1] + rec_mid[3], rec_mid[0]:rec_mid[0] + rec_mid[2]] * (1 - mask) + mid_img_rect * mask
    
#=====================================================================
"""
Face morphing. Main function.
"""
def main(index, alpha = 0.5):
    #file name
    face1_file = "./face_data/01-1m.bmp"
    face1_data_file = "./face_data/01-1m.txt"
    face2_file = "./face_data/05-1m.bmp"
    face2_data_file = "./face_data/05-1m.txt"

    # mean shape
    dir = "./face_data/data/"
    mean_shape = mean_face_shape(dir)
    mean_face_img = mean_face(dir)
    
    # set figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 24
    fig_size[1] = 16


    # read data
    """
    Data type:
        path_id
        point_type
        x_rel
        y_rel
        point_id
        connectsFrom
        connectsTo
    """
    path_id1, point_type1, pointSet1, point_id1, connectsFrom1, connectsTo1 = get_data(face1_data_file)
    path_id2, point_type2, pointSet2, point_id2, connectsFrom2, connectsTo2 = get_data(face2_data_file)
    
    # num of points
    num_pts = len(path_id1)

    # Delaunay triangulation
    pointSet1 = pointSet1.astype(dtype='int32')
    pointSet2 = pointSet2.astype(dtype='int32')
    del_triangle1 = Delaunay(pointSet1)
    del_triangle2 = Delaunay(pointSet2)

    """
    add few more points for the background
    four corners of the image
    four middle points of the image edges
    """

    # read image file
    img1 = cv2.imread(face1_file)
    img2 = cv2.imread(face2_file)
    img1 = img1.astype(dtype='float32')
    img2 = img2.astype(dtype='float32')

    """
    Slight difference in pixel ordering in OpenCV and Matplotlib.
    OpenCV follows BGR order, while matplotlib likely follows RGB order.
    So the image need to re-order to show correctly
    When using cv2.imshow, no need to re-order
    """
    
    # image in between
    # point_mid = alpha * point_in_src + (1 - alpha) * point_in_dst
    pointSet_mid = []
    beta = 1 - alpha
    for i in range(num_pts):
        x_mid = alpha * pointSet1[i, 0] + beta * pointSet2[i, 0]
        y_mid = alpha * pointSet1[i, 1] + beta * pointSet2[i, 1]
        pointSet_mid.append((x_mid, y_mid))
    pointSet_mid = np.asarray(pointSet_mid)
    
    # morphing image
    # create an empty image
    img_mid = np.zeros(img1.shape, dtype = img1.dtype)
    pointSet_mid = mean_shape
    for file in listdir(dir):
        # "*.asf" file is what we need
        if (".bmp" in file):
            # read image
            img1 = cv2.imread(dir + file)
            # compute the mean face
            #mean_face = np.float32(face_img) + mean_face
            # morphing: one triangle each time
            for tri in del_triangle1.simplices:
                # get the coordinates from the source image
                tri_src = pointSet1[tri].astype(dtype='float32')
                # get the coordinates from the destination image
                tri_dst = pointSet2[tri].astype(dtype='float32')
                # get the coordinates from the destination image
                tri_mid = pointSet_mid[tri].astype(dtype='float32')
                morphing(img1, img2, img_mid, tri_src, tri_dst, tri_mid, alpha)
    
    # =====================================================================
    """
    Display the image with matplotlib
    Something wrong when display the image from opencv
    Not used anymore.   
    """
    #f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
    #plt.subplot(121), plt.imshow(img1), plt.title('Src')
    #plt.scatter([270, 400], [150, 150])
    #face1 = mask_tri1 + face1
    #plt.subplot(122), plt.imshow(img_mid), plt.title('In-between')
    #plt.scatter(pointSet1[:, 0], pointSet1[:, 1])
    #plt.triplot(pointSet1[:,0], pointSet1[:,1], del_triangle1.simplices.copy())
    #plt.show()
    #plt.show()
    #rec_src = cv2.boundingRect(pointSet1[tri].astype(dtype='float32'))
    #print (rec_src)
    # =====================================================================
    output_file = "img{}.jpg".format(index)
    #cv2.imshow("Morphed Face", np.uint8(img_mid))
    cv2.imwrite(output_file, np.uint8(img_mid))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#=====================================================================
"""
Read face data from a certain directory and compute the average face
File name: "*.asf", ignore those line starting with "#"
"""
def mean_face_shape(dir):
    # initialize
    mean_shape = np.zeros([58, 2])
    # counting the number of files
    count = 0
    # scan all the file names in dir
    for file in listdir(dir):
        # "*.asf" file is what we need
        if (".asf" in file):
            # read points
            path_id, point_type, pointSet, point_id, connectsFrom, connectsTo = read_asf(dir + file)
            # compute the mean face
            mean_shape = pointSet + mean_shape
            # count increment
            count = count + 1
    
    # average
    mean_shape = mean_shape / count
    return mean_shape

    #=====================================================================
"""
Read face image from a certain directory and compute the average texture
File name: "*.bmp"
"""
def mean_face(dir):
    # initialize
    mean_face = np.zeros([480, 640, 3])
    count = 0
    # scan all the file names in dir
    for file in listdir(dir):
        # "*.asf" file is what we need
        if (".bmp" in file):
            # read image
            face_img = cv2.imread(dir + file)
            # compute the mean face
            mean_face = np.float32(face_img) + mean_face
            count = count + 1
    # average
    mean_face = mean_face / count
    return mean_face

#=====================================================================
    
# run
if __name__ == "__main__":
    rate = 1 / 50
    for i in range(1):
        rate = rate + 1 / 50
        main(i, alpha = rate)
