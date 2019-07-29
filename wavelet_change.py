import pywt, cv2
import numpy as np

def img_normalization(src_img):
  return ((src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img)))

def merge_images(cA, cH_V_D):
    """numpy.array を４つ(左上、(右上、左下、右下))連結させる"""
    cH, cV, cD = cH_V_D
    cH = img_normalization(cH) # 外してもok
    cV = img_normalization(cV) # 外してもok
    cD = img_normalization(cD) # 外してもok
    cA = cA[0:cH.shape[0], 0:cV.shape[1]] # 元画像が2の累乗でない場合、端数ができることがあるので、サイズを合わせる。小さい方に合わせます。
    return np.vstack((np.hstack((cA,cH)), np.hstack((cV, cD)))) # 左上、右上、左下、右下、で画素をくっつける

def coeffs_visualization(cof):
    norm_cof0 = cof[0]
    norm_cof0 = img_normalization(norm_cof0) # 外してもok
    merge = norm_cof0
    for i in range(1, len(cof)):
        merge = merge_images(merge, cof[i])  # ４つの画像を合わせていく
    cv2.imshow('', merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def wavelet_transform_for_image(src_image, level, M_WAVELET="db1", mode="sym"):
    data = src_image.astype(np.float64)
    coeffs = pywt.wavedec2(data, M_WAVELET, level=level, mode=mode)
    return coeffs

if __name__ == "__main__":

    filename = 'img/maho00.jpg'
    LEVEL = 3
    MOTHER_WAVELET = "db1"

    im = cv2.imread(filename)

    print('LEVEL :', LEVEL)
    print('MOTHER_WAVELET', MOTHER_WAVELET)
    print('original image size: ', im.shape)

    """
    各BGRチャネル毎に変換
    cv2.imreadはB,G,Rの順番で画像を吐き出すので注意
    """
    B = 0
    G = 1
    R = 2
    coeffs_B = wavelet_transform_for_image(im[:, :, B], LEVEL, M_WAVELET=MOTHER_WAVELET)
    coeffs_G = wavelet_transform_for_image(im[:, :, G], LEVEL, M_WAVELET=MOTHER_WAVELET)
    coeffs_R = wavelet_transform_for_image(im[:, :, R], LEVEL, M_WAVELET=MOTHER_WAVELET)

    coeffs_visualization(coeffs_B)
    # coeffs_visualization(coeffs_G)
    # coeffs_visualization(coeffs_R)