import json
import numpy as np
import os
import cv2
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import time
from sender import UDPSender

# np.set_printoptions(threshold=np.inf)

# 生成middle stick filter算子
def generate_middle_stick_filter(L):
    if L % 2 == 0 or L < 3:
        raise (ValueError("L should be an odd number and bigger than 3"))
    # 初始filte数量,后续filter数量为2l-1,可通过旋转,镜像操作生成
    filter_num = int((L - 1) / 2 + 1)

    result = []

    r = int((L - 1) / 2)
    for i in range(0, filter_num):
        tmp = np.zeros((L, L))
        tmp[r][r] = 1
        for m in range(0,r):
            for n in range(0, r + 1):
                if n == int(m + i * (L - 2 * m)/ L + 0.5):
                    tmp[m][n] = 1
                    tmp[L - m -1][L - n -1] = 1
        result.append(tmp)

    for i in range(0, r):
        result.append(np.flip(result[r-i-1], 0))

    for i in range(1, L-1):
         result.append(result[L-1-i].transpose(1, 0))

    return np.asarray(result)

# 生成middle minus left 算子
def generate_middle_minus_left_filter(m_filters, S):
    L =int( m_filters.shape[0]/2 +1)
    print(L)
    result = m_filters.copy()
    for i in range(result.shape[0]):
        for m in range(0, L):
            for n in range(0, L):
                if result[i][m][n] == 1:
                    if i < L:
                        if n - S >= 0:
                            result[i][m][n - S] = -1
                        else:
                            result[i][m][n]=0
                    else:
                        if m - S >= 0:
                            result[i][m - S][n] = -1
                        else:
                            result[i][m][n]=0
    return result

# 生成middle minus right 算子
def generate_middle_minus_right_filter(m_filters, S):
    L =int( m_filters.shape[0]/2 +1)
    print(L)
    result = m_filters.copy()
    for i in range(result.shape[0]):
        for m in range(0, L):
            for n in range(0, L):
                if result[i][m][n] == 1:
                    if i < L:
                        if n + S < L:
                            result[i][m][n + S] = -1
                        else:
                            result[i][m][n]=0
                    else:
                        if m + S < L:
                            result[i][m + S][n] = -1
                        else:
                            result[i][m][n]=0
    return result

def my_convolution_cascade(l_kernels, r_kernels, val_kernels,data,index,need_orientation,k_val,max_index_arr):
    k_nums = l_kernels.shape[0]
    L=int(1+(k_nums/2))

    img_new = []

    square_data = np.multiply(data,data)
    for k in range(k_nums):
        tmp_img_l = cv2.filter2D(data,-1,l_kernels[k])
        tmp_img_r = cv2.filter2D(data, -1, r_kernels[k])

        tmp_img = np.zeros(tmp_img_l.shape+np.asarray([0,0]).shape)
        tmp_img[:, :, 0] = tmp_img_l
        tmp_img[:, :, 1] = tmp_img_r

        tmp_img = np.max(tmp_img, axis=2)


        EI2i_arr = cv2.filter2D(square_data, -1, val_kernels[k]).astype(np.int32)/L
        tmp = (cv2.filter2D(data,-1,val_kernels[k])).astype(np.int32)/L
        EIi2_arr = np.multiply(tmp,tmp)

        var_arr = EI2i_arr - EIi2_arr
        var_arr[var_arr <0] = 0

        var_arr = np.sqrt(var_arr).astype(np.int32)

        tmp_img = tmp_img-k_val*var_arr

        img_new .append(tmp_img)


    img_new=np.asarray(img_new)
    img_new = img_new.transpose(1, 2, 0)


    img_new = np.max(img_new, axis=2)

    square_data = np.multiply(img_new, img_new)
    img_result = []
    for k in range(k_nums):
        tmp_img_l = cv2.filter2D(img_new , -1, l_kernels[k])
        tmp_img_r = cv2.filter2D(img_new , -1, r_kernels[k])

        tmp_img = np.zeros(tmp_img_l.shape + np.asarray([0, 0]).shape)
        tmp_img[:, :, 0] = tmp_img_l
        tmp_img[:, :, 1] = tmp_img_r

        tmp_img = np.min(tmp_img, axis=2)

        EI2i_arr = cv2.filter2D(square_data, -1, val_kernels[k]).astype(np.int32) / L
        tmp = (cv2.filter2D(img_new, -1, val_kernels[k])).astype(np.int32) / L
        EIi2_arr = np.multiply(tmp, tmp)

        var_arr = EI2i_arr - EIi2_arr
        var_arr[var_arr < 0] = 0

        var_arr = np.sqrt(var_arr).astype(np.int32)

        tmp_img = tmp_img - k_val * var_arr

        img_result.append(tmp_img)

    img_result  = np.asarray(img_result )
    img_result = img_result.transpose(1,2,0)
    if need_orientation == 1:
        max_index_arr[:,:,index] = np.argmax(img_result, axis=2)

    result = np.max(img_result, axis=2)
    result[result<0]=0
    return np.array(result)


# 可以用来增强右肺斜裂,实际也可以用来提取左肺斜裂,不记录方向
# data: HU ROI
def right_lung_fissure_enhance(data,l,w,k):
    # 初始化UDP发送端
    sender = UDPSender("127.0.0.1", 8001)

    np.seterr(divide='ignore', invalid='ignore')
    # 中间算子
    mid_kernels = generate_middle_stick_filter(l)
    # for i in mid_kernels:
    #     print(i)
    minus_left_kernels = generate_middle_minus_left_filter(mid_kernels, w)

    minus_right_kernels = generate_middle_minus_right_filter(mid_kernels, w)

    # 这里为了实现简单,选择方向写死了,实际取范围应该根据l的值进行调整
    mid_kernels_end_view = mid_kernels[5:15]
    minus_left_kernels_end_view = minus_left_kernels[5:15]
    minus_right_kernels_end_view = minus_right_kernels[5:15]

    images = data

    # 用来存三视图分别的滤波结果,为了提升效率,俯视图不再做处理
    three_view_result = np.zeros(images.shape + np.asarray([0, 0]).shape).astype(np.uint32)



    # 发送图片张数
    nums = images.shape[0]*4

    sender.send_something(str(nums).encode())

    sender.encode_send_images(images)


    # test
    # string_result = sender.get_encode_result(4)
    # result = sender.decode_and_save(string_result);

    # 接受图片并解析
    string_result_front = sender.get_encode_result(images.shape[1]*4)
    string_result_end = sender.get_encode_result(images.shape[2]*4)

    # front_view_result = sender.decode_and_save(string_result_front)
    front_view_result = sender.decode_and_save(string_result_front)
    end_view_result = sender.decode_and_save(string_result_end)

    # np.set_printoptions(threshold=np.inf)
    # with open('f1', 'wt') as f1:
    #     print(front_view_result, file=f1)


    # gapi
    # pipe = pipeline.Pipeline(minus_left_kernels, minus_right_kernels, mid_kernels)
    # pipe.graph()
    
    # 正视图结果
    # front_view_result = []
    # for i in range(images.shape[1]):
    #     # for i in range(5):
    #     # print(i)
    #     # mark_arr.append(my_convolution1(test_l,test_r, images[i]))
    #     front_view_result.append(
    #         my_convolution_cascade(minus_left_kernels, minus_right_kernels, mid_kernels, images[:, i, :], i, 0,k,None))
    #         # pipeline.my_convolution_cascade_gapi(minus_left_kernels, images[:, i, :], i, 0, k, None, pipe))
    front_view_result = np.asarray(front_view_result).astype(np.uint32).transpose(1, 0, 2)
    three_view_result[:, :, :, 0] = front_view_result


    # 侧视图结果
    # end_view_result = []
    # for i in range(images.shape[2]):
    #     end_view_result.append(
    #         my_convolution_cascade(minus_left_kernels_end_view, minus_right_kernels_end_view, mid_kernels_end_view, images[:, :, i], i, 0,k,None))
    end_view_result = np.asarray(end_view_result).astype(np.uint32).transpose(1, 2, 0)
    three_view_result[:, :, :, 1] = end_view_result

    # vertical_view_result += end_view_result

    three_view_result = np.sort(three_view_result, axis=3)
    accumulate_result = np.add(three_view_result[:, :, :, 0],
                               three_view_result[:, :, :, 1])

    accumulate_result = np.multiply(three_view_result[:, :, :, 0], accumulate_result)

    accumulate_result = np.floor_divide(accumulate_result, three_view_result[:, :, :, 1])
    accumulate_result = accumulate_result.astype(np.uint32)
    return accumulate_result



# 左肺肺裂增强
def left_lung_fissure_enhance(data,l,w,k):
    sender = UDPSender("127.0.0.1", 8001)


    np.seterr(divide='ignore', invalid='ignore')
    # 中间算子
    mid_kernels = generate_middle_stick_filter(l)
    # for i in mid_kernels:
    #     print(i)
    minus_left_kernels = generate_middle_minus_left_filter(mid_kernels, w)

    minus_right_kernels = generate_middle_minus_right_filter(mid_kernels, w)

    max_index_arr = np.zeros(data.shape).astype(np.uint8)
    print(max_index_arr.shape)
    images = data

    # 用来存三视图分别的滤波结果,为了提升效率,俯视图不再做处理
    three_view_result = np.zeros(images.shape + np.asarray([0, 0]).shape).astype(np.uint32)

    nums = images.shape[0]*4
    sender.send_something(str(nums).encode())
    # 发送图片
    sender.encode_send_images(images)



    # 接受图片并解析
    string_result_front = sender.get_encode_result(images.shape[1]*4)
    string_result_end = sender.get_encode_result(images.shape[2]*4)
    string_max_index_arr = sender.get_encode_result(images.shape[0])

    front_view_result = sender.decode_and_save(string_result_front)
    end_view_result = sender.decode_and_save(string_result_end)
    max_index_arr = np.array(sender.decode_and_save_2(string_max_index_arr))

    print("max_index_arr shape:")
    print(max_index_arr.shape)

    # gapi
    # pipe = pipeline.Pipeline(minus_left_kernels, minus_right_kernels, mid_kernels)
    # pipe.graph()
    
    # 正视图结果
    # front_view_result = []
    # for i in range(images.shape[1]):
    #     # for i in range(5):
    #     # print(i)
    #     # mark_arr.append(my_convolution1(test_l,test_r, images[i]))
    #     front_view_result.append(
    #         my_convolution_cascade(minus_left_kernels, minus_right_kernels, mid_kernels, images[:, i, :], i, 0,k,None))
            # pipeline.my_convolution_cascade_gapi(minus_left_kernels, images[:, i, :], i, 0, k, None, pipe))
    front_view_result = np.asarray(front_view_result).astype(np.uint32).transpose(1, 0, 2)
    three_view_result[:, :, :, 0] = front_view_result


    # 侧视图结果
    # end_view_result = []
    # for i in range(images.shape[2]):
    #     end_view_result.append(
    #         my_convolution_cascade(minus_left_kernels, minus_right_kernels, mid_kernels, images[:, :, i], i, 1,k,max_index_arr))
    end_view_result = np.asarray(end_view_result).astype(np.uint32).transpose(1, 2, 0)
    three_view_result[:, :, :, 1] = end_view_result

    # vertical_view_result += end_view_result

    three_view_result = np.sort(three_view_result, axis=3)
    accumulate_result = np.add(three_view_result[:, :, :, 0],
                               three_view_result[:, :, :, 1])

    accumulate_result = np.multiply(three_view_result[:, :, :, 0], accumulate_result)

    accumulate_result = np.floor_divide(accumulate_result, three_view_result[:, :, :, 1])
    accumulate_result = accumulate_result.astype(np.uint32)
    return accumulate_result,max_index_arr

def r_fissure_enhance(data):
    # f = open('right_lung_roi.pkl', 'wb')
    # pickle.dump(data, f)
    # f.close()
    fissure_enhancement_result = right_lung_fissure_enhance(data, 11, 2, 7)
    return fissure_enhancement_result

def l_fissure_enhance(data):
    # f = open('left_lung_roi.pkl', 'wb')
    # pickle.dump(data, f)
    # f.close()
    fissure_enhancement_result,fissure_orientation = left_lung_fissure_enhance(data, 11, 2, 7)
    return fissure_enhancement_result,fissure_orientation