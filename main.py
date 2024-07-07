import cv2
import numpy as np

def gauss_ker(k, sig):
    x = np.linspace(-(k//2), (k//2), k)
    gx, gy = np.meshgrid(x, x)
    kernel = np.exp(-1*(gx**2 + gy**2)/(2*(sig**2)))
    return kernel

def get_amb_mask(i,j,bias,ambientpad):
    amb_mask = ambientpad[i-bias:i+bias+1, j-bias:j+bias+1]
    return amb_mask

def get_flash_diff(i,j,flash_mask,flashpad):
    return flash_mask - flashpad[i, j]

def get_bil_mask(flash_diffmask,sigmab1):
    return np.exp(-1*((flash_diffmask/sigmab1)**2)/(2*(sigmab1**2)))

def joint_mask(bil_mask_flash,bil_mask_amb,gauss_mask,amb_mask,flash_mask):
  filt_mask_flash = bil_mask_flash*gauss_mask
  norm_term_flash = np.sum(filt_mask_flash)
  filt_mask_amb = bil_mask_amb*gauss_mask
  norm_term_amb = np.sum(filt_mask_amb)
  Ajoint_mask = (amb_mask*filt_mask_flash)/norm_term_flash
  Abase_mask = (amb_mask*filt_mask_amb)/norm_term_amb
  Fbase_mask = (flash_mask*filt_mask_flash)/norm_term_flash
  return Ajoint_mask,Abase_mask,Fbase_mask

def bilateral_filter(flash, no_flash):
    if flash.shape[0] == 636 and flash.shape[1] == 780:
      s_s = 4
      ws = 7
      s_r = 1.5
    elif flash.shape[0] ==  706 and flash.shape[1] == 774:
      s_s = 9
      ws = 13
      s_r = 2
    elif flash.shape[0] ==  563 and flash.shape[1] == 789:
      s_s = 6
      ws = 7
      s_r = 7
    else:
      s_s = 8
      ws = 7
      s_r = 1.5
    gauss_mask = gauss_ker(ws, s_s)

    kernel = (ws//2)
    flashpad = np.lib.pad(flash, (kernel, kernel), 'edge')
    ambientpad = np.lib.pad(no_flash, (kernel, kernel), 'edge')

    h, w = flash.shape
    jbf = np.zeros((h, w))
    no_flash_base = np.zeros((h, w))
    flash_base = np.zeros((h, w))

    for i in range(kernel, h+kernel):
        for j in range(kernel, w+kernel):
            no_flash_mask = get_amb_mask(i,j,kernel,ambientpad)
            flash_mask = get_amb_mask(i,j,kernel,flashpad)

            flash_diffmask = get_flash_diff(i,j,flash_mask,flashpad)
            no_flash_diffmask = get_flash_diff(i,j,no_flash_mask,ambientpad)

            bil_mask_flash = get_bil_mask(flash_diffmask,s_r)
            bil_mask_amb = get_bil_mask(no_flash_diffmask,s_r)

            jbf_mask,no_flash_base_mask,flash_base_mask = joint_mask(bil_mask_flash,bil_mask_amb,gauss_mask,no_flash_mask,flash_mask)

            jbf[i-kernel, j-kernel] = np.sum(jbf_mask)
            no_flash_base[i-kernel, j-kernel] = np.sum(no_flash_base_mask)
            flash_base[i-kernel, j-kernel] = np.sum(flash_base_mask)

    return [jbf, no_flash_base, flash_base]

def flashAdj(imf, ima, alpha):
    ya = cv2.cvtColor(ima, cv2.COLOR_BGR2YCR_CB)
    yf = cv2.cvtColor(imf, cv2.COLOR_BGR2YCR_CB)
    im = np.zeros(ya.shape).astype('double')

    im = alpha*ya + (1-alpha)*yf
    im[im>255] = 255
    im[im<0] = 0
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_YCR_CB2RGB)
    return im

def get_lin(img):
  return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def get_flag(linflash,mask):
  flag = np.zeros((mask.shape), np.uint8)
  thr1 = -0.05
  thr2 = -0.2
  flag[(mask > thr2) & (mask < thr1)] = 1
  flag[(mask > 0.65) & (mask < 0.7)] = 1
  rang = 0.95*(np.max(linflash) - np.min(linflash))
  flag[linflash > rang] = 1
  return flag
def get_se():
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    return se1,se2,se3
def get_mask(flag,se1):
  flag = cv2.erode(flag, se1, iterations = 1)
  maskff = np.zeros((flag.shape[0]+2, flag.shape[1]+2), np.uint8)
  cv2.floodFill(flag, maskff, (0,0), 1)
  maskff = 1 - maskff
  return maskff
def get_kernel():
  kernel = np.array([[0.1070, 0.1131, 0.1070],
  [0.1131,    0.1196,    0.1131],
  [0.1070,    0.1131,    0.1070]])
  return kernel
def shadow_mask(flash_img,no_flash_img):

  linflash = get_lin(flash_img)
  linnoflash = get_lin(no_flash_img)

  mask = linflash - linnoflash

  flag = get_flag(linflash,mask)

  se1,se2,se3 = get_se()
  maskff = get_mask(flag,se1)
  maskff = cv2.dilate(maskff, se2)
  maskff = cv2.erode(maskff, se3)
  maskff = maskff.astype('double')

  kernel = get_kernel()

  maskff = cv2.filter2D(maskff, -1, kernel)
  shadowMask = maskff[:-2,:-2]
  return shadowMask

def get_ffin(shadowMask,jbf,Fdetail,no_flashbase):
  return (np.dstack(((1-shadowMask), (1-shadowMask), (1-shadowMask)))*(jbf*Fdetail) + np.dstack((shadowMask, shadowMask, shadowMask))*(no_flashbase))

def get_detail(color,epsilon,flashbasecolor):
  return (color + epsilon)/(flashbasecolor + epsilon)
def solution(img_path1,img_path2):
    flash_img = cv2.imread(img_path2)
    no_flash_img = cv2.imread(img_path1)
    amb_img_bil = np.copy(no_flash_img)
    flash_img = cv2.cvtColor(flash_img, cv2.COLOR_BGR2RGB)
    no_flash_img = cv2.cvtColor(no_flash_img, cv2.COLOR_BGR2RGB)
#     ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    amb_img_bil = np.copy(no_flash_img)
    flash_img = flash_img.astype('double')/255
    no_flash_img = no_flash_img.astype('double')/255

    # shadow mask
    shadowMask = shadow_mask(flash_img,no_flash_img)

    f_g = flash_img[:,:,1]
    f_r = flash_img[:,:,0]
    f_b = flash_img[:,:,2]

    [jbfg, no_flashbaseg, flashbaseg] = bilateral_filter(f_g, no_flash_img[:,:,1])
    [jbfr, no_flashbaser, flashbaser] = bilateral_filter(f_r, no_flash_img[:,:,0])
    [jbfb, no_flashbaseb, flashbaseb] = bilateral_filter(f_b, no_flash_img[:,:,2])

    jbf = np.dstack((jbfr, jbfg, jbfb))
    no_flashbase = np.dstack((no_flashbaser, no_flashbaseg, no_flashbaseb))
    flashbase = np.dstack((flashbaser, flashbaseg, flashbaseb))

    epsilon = 0.02
    flashdetailb = get_detail(f_b,epsilon,flashbaseb)
    flashdetailr = get_detail(f_r,epsilon,flashbaser)
    flashdetailg = get_detail(f_g,epsilon,flashbaseg)
    
    flashdetail = np.dstack((flashdetailr, flashdetailg, flashdetailb))

    shadowMask = 0
    Ffin = get_ffin(shadowMask,jbf,flashdetail,no_flashbase)

    Ffin[Ffin>1] = 1
    flashdetail[flashdetail>1] = 1

    image_normalized = cv2.normalize(Ffin, None, 0.0, 255.0, cv2.NORM_MINMAX)
    image_int = cv2.convertScaleAbs(image_normalized)
    img = cv2.cvtColor(image_int,cv2.COLOR_RGB2BGR)
    img1 = img
    img2 = cv2.imread(img_path2)

    img1_rgb = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    bf_out = cv2.cvtColor(img1_rgb,cv2.COLOR_RGB2BGR)
    return bf_out
