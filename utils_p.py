import math
import cv2
import torch
import numpy as np
from math import *
from scipy.special import iv

class AverageMeter(object):
    #Computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_angular_loss(vec1,vec2):
    safe_v = 0.999999
    illum_normalized1 = torch.nn.functional.normalize(vec1,dim=1)
    # print(illum_normalized1)
    illum_normalized2 = torch.nn.functional.normalize(vec2,dim=1)
    # illum_normalized2 = vec2
    # print(illum_normalized2)
    dot = torch.sum(illum_normalized1*illum_normalized2,dim=1)
    # print(dot)
    dot = torch.clamp(dot, -safe_v, safe_v)
    # print(dot)
    angle = torch.acos(dot)*(180/math.pi)
    # print(angle)
    loss = torch.mean(angle)
    return loss






def get_angular_loss_ang(vec1,vec2):
  safe_v = 0.999999
    # illum_normalized1 = torch.nn.functional.normalize(vec1,dim=1)
    # print(illum_normalized1)
    # illum_normalized2 = torch.nn.functional.normalize(vec2,dim=1)
    # illum_normalized2 = vec2
    # print(illum_normalized2)
  dot = torch.sum(vec1*vec2,dim=1)
  # print(dot)
  dot = torch.clamp(dot, -safe_v, safe_v)
  # print(dot)
  angle = torch.acos(dot)*(180/math.pi)
  # print(angle)
  loss = torch.mean(angle)
  return loss



def get_awb_pic(img,pred):
    # pred = torch.sum(pred, (2))
    print('pred_shape',pred.shape)
    pred = torch.nn.functional.normalize(pred,dim=1)
    pred = pred.detach().cpu().numpy()
    pred = np.array(pred,'float32')
    pred = pred.squeeze(0).transpose(1,0)
    # pred = normalize(pred)
    pred_s1 = cv2.resize(pred.reshape((16,16,3)),(256,256))
    pred[:,0] = pred[:,0]/pred[:,1]
    pred[:,2] = pred[:,2]/pred[:,1]
    pred[:,1] = pred[:,1]/pred[:,1]
    # print(pred.shape)
    # print('img_min:',np.min(img),"img_max:",np.max(img))
    img = np.clip(img,0,1)
    # print(np.max(img))
    # pred_s2 = pred.reshape(64,64,3)
    pred_s2 = cv2.resize(pred.reshape((16,16,3)),(256,256))
    img_wb = np.clip(img/pred_s2,0,1)**(1/2.2)*255
    img_wb = np.uint8(img_wb)
    # print(np.max(img_wb))
    return img_wb,pred_s1

def get_awb_pic_single(img,pred,pred_map):
    pred = torch.sum(pred, (2))
    pred = torch.nn.functional.normalize(pred,dim=1)
    pred = pred.squeeze(0)
    pred = pred.detach().cpu().numpy()
    pred = np.array(pred,'float32')
    pred_pmap = pred*np.ones_like(pred_map)
    # pmap_t = torch.from_numpy(pred_pmap.copy())
    # pred_mapt
    pred_map = pred_map/pred_pmap
    weights = pred_map
    # weights = pred_map/pred_pmap
    # pred = normalize(pred)
    pred = pred/pred[1]
    # pred = pred.transpose(1,0)
    # print(pred.shape)
    # print('img_min:',np.min(img),"img_max:",np.max(img))
    img = np.clip(img,0,1)
    # print(np.max(img))
    # pred = cv2.resize(pred.reshape(64,64,3),(256,256))
    img_wb = np.clip(img/pred,0,1)**(1/2.2)*255
    img_wb = np.uint8(img_wb)
    # print(np.max(img_wb))
    return img_wb,pred_map,weights

def get_gt_pic(img,pred):
    pred = torch.nn.functional.normalize(pred,dim=1)
    pred = pred.squeeze(0)
    pred = pred.detach().cpu().numpy()
    pred = np.array(pred,'float32')
    pred = normalize(pred)
    pred = pred/pred[1]
    print(pred)
    print('img_min:',np.min(img),"img_max:",np.max(img))
    img = np.clip(img/np.max(img),0,1)
    print(np.max(img))
    img_wb = np.clip(img/pred,0,1)**(1/2.2)*255
    img_wb = np.uint8(img_wb)
    print(np.max(img_wb))
    return img_wb



def correct_image_nolinear(img,ill): 
    #nolinear img, linear ill , return non-linear img
    nonlinear_ill = torch.pow(ill,1.0/2.2)
    correct = nonlinear_ill.unsqueeze(2).unsqueeze(3)*torch.sqrt(torch.Tensor([3])).cuda()
    correc_img = torch.div(img,correct+1e-10) 
    img_max = torch.max(torch.max(torch.max(correc_img,dim=1)[0],dim=1)[0],dim=1)[0]+1e-10
    img_max = img_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    img_normalize = torch.div(correc_img,img_max)
    return img_normalize 


def evaluate(errors):
    errors = sorted(errors)
    
    def g(f):
        return np.percentile(errors,f*100)
    median = g(0.5)        
    mean = np.mean(errors)
    trimean = 0.25*(g(0.25)+2*g(0.5)+g(0.75))
    bst25 = np.mean(errors[:int(0.25*len(errors))])
    wst25 = np.mean(errors[int(0.75*len(errors)):])
    pct95 = g(0.95)
    return mean,median,trimean,bst25,wst25,pct95

def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) / 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]

  # Find the size of the new image
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))

  # We require a translation matrix to keep the image centred
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result

def largest_rotated_rect(w, h, angle):
  """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

  quadrant = int(math.floor(angle / (math.pi / 2))) & 3
  sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
  alpha = (sign_alpha % math.pi + math.pi) % math.pi

  bb_w = w * math.cos(alpha) + h * math.sin(alpha)
  bb_h = w * math.sin(alpha) + h * math.cos(alpha)

  gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

  delta = math.pi - alpha - gamma

  length = h if (w < h) else w

  d = length * math.cos(alpha)
  a = d * math.sin(alpha) / math.sin(delta)

  y = a * math.cos(gamma)
  x = y * math.tan(gamma)

  return (bb_w - 2 * x, bb_h - 2 * y)

def crop_around_center(image, width, height):
  """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

  if (width > image_size[0]):
    width = image_size[0]

  if (height > image_size[1]):
    height = image_size[1]

  x1 = int(image_center[0] - width * 0.5)
  x2 = int(image_center[0] + width * 0.5)
  y1 = int(image_center[1] - height * 0.5)
  y2 = int(image_center[1] + height * 0.5)

  return image[y1:y2, x1:x2]


def rotate_and_crop(image, angle):
  image_width, image_height = image.shape[:2]
  image_rotated = rotate_image(image, angle)
  image_rotated_cropped = crop_around_center(image_rotated,
                                             *largest_rotated_rect(
                                                 image_width, image_height,
                                                 math.radians(angle)))
  return image_rotated_cropped
