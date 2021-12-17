#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <ctime>

using namespace std;
using namespace cv;

vector<Mat> filter_loop(vector<Mat> l_kernels, vector<Mat> r_kernels, vector<Mat> val_kernels, Mat data, int k_val, int dtype);

Mat mat_vector_max(vector<Mat> img_new);

Mat my_convolution_cascade(vector<Mat> l_kernels, vector<Mat> r_kernels, vector<Mat> val_kernels, Mat data, int index, bool need_orientation, int k_val, vector<Mat>* max_index_arr);

vector<Mat> generate_middle_stick_filter(int L);

vector<Mat> generate_middle_minus_left_filter(vector<Mat> m_filters, int S);

vector<Mat> generate_middle_minus_right_filter(vector<Mat> m_filters, int S);

vector<vector<Mat>> lung_fissure_enhance(vector<Mat> images, int l, int w, int k, bool is_left);

vector<vector<Mat>> filterTest(vector<Mat> data, bool is_left);