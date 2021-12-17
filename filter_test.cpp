#include "filter_test.h"

vector<Mat> filter_loop(vector<Mat> l_kernels, vector<Mat> r_kernels, vector<Mat> val_kernels, Mat data, int k_val, int dtype)
{
	int k_num = l_kernels.size();
	int L = 1 + (k_num / 2);

	vector<Mat> img_new;
	
	Mat square_data = Mat::zeros(data.size(), CV_32F);
	square_data = data.mul(data);

	cuda::GpuMat data_gpu(data.size(), data.type()); //
	cuda::GpuMat square_data_gpu(square_data.size(), square_data.type());
	cuda::GpuMat tmp_img_l_gpu(data.size(), data.type()), tmp_img_r_gpu(data.size(), data.type());
	cuda::GpuMat EI2i_arr_gpu(square_data.size(), square_data.type()), tmp_gpu(data.size(), data.type());
	data_gpu.upload(data);
	square_data_gpu.upload(square_data);

	for (int k = 0; k < k_num; ++k)
	{
		Ptr<cuda::Filter> l_filter = cuda::createLinearFilter(CV_32F, CV_32F, l_kernels[k]);
		Ptr<cuda::Filter> r_filter = cuda::createLinearFilter(CV_32F, CV_32F, r_kernels[k]);
		Ptr<cuda::Filter> val_filter = cuda::createLinearFilter(CV_32F, CV_32F, val_kernels[k]);

		l_filter->apply(data_gpu, tmp_img_l_gpu);
		r_filter->apply(data_gpu, tmp_img_r_gpu);
	
		Mat tmp_img;
		if (dtype == 0)
		{
			cuda::max(tmp_img_l_gpu, tmp_img_r_gpu, tmp_img);
		}
		else if (dtype == 1)
		{
			cuda::min(tmp_img_l_gpu, tmp_img_r_gpu, tmp_img);
		}
		
		Mat EI2i_arr, tmp;
		val_filter->apply(square_data_gpu, EI2i_arr_gpu);
		val_filter->apply(data_gpu, tmp_gpu);
		EI2i_arr_gpu.download(EI2i_arr);
		tmp_gpu.download(tmp);

		EI2i_arr /= L;
		tmp /= L;
		Mat EIi2_arr = tmp.mul(tmp);

		Mat var_arr = EI2i_arr - EIi2_arr;
		for (int i = 0; i < var_arr.rows; ++i)
			for (int j = 0; j < var_arr.cols; ++j)
			{
				if (var_arr.at<float>(i, j) < 0)
					var_arr.at<float>(i, j) = 0;
			}

		sqrt(var_arr, var_arr);
		tmp_img -= k_val * var_arr;

		img_new.push_back(tmp_img);
	} 

	return img_new;
}

Mat mat_vector_max(vector<Mat> img_new)
{
	Mat new_img = img_new.front().clone();
	for (int i = 0; i < new_img.rows; ++i)
		for (int j = 0; j < new_img.cols; ++j)
		{
			float tmp = new_img.at<float>(i, j);
			for (vector<Mat>::iterator it = img_new.begin(); it != img_new.end(); ++it)
			{
				if (tmp < it->at<float>(i, j))
					tmp = it->at<float>(i, j);
			}
			new_img.at<float>(i, j) = tmp;
		}
	return new_img;
}

Mat my_convolution_cascade(vector<Mat> l_kernels, vector<Mat> r_kernels, vector<Mat> val_kernels, Mat data, int index, bool need_orientation, int k_val, vector<Mat>* max_index_arr)
{
	vector<Mat> img_new = filter_loop(l_kernels, r_kernels, val_kernels, data, k_val, 0);

	Mat new_img = mat_vector_max(img_new);

	vector<Mat> img_result = filter_loop(l_kernels, r_kernels, val_kernels, new_img, k_val, 1);

	if (need_orientation)
	{
		Mat new_img = img_result.front().clone();
		for (int i = 0; i < new_img.rows; ++i)
			for (int j = 0; j < new_img.cols; ++j)
			{
				float tmp = new_img.at<float>(i, j);
				int arg = 0, k = 0;
				for (vector<Mat>::iterator it = img_result.begin(); it != img_result.end(); ++it, ++k)
				{
					if (tmp < it->at<float>(i, j))
					{
						tmp = it->at<float>(i, j);
						arg = k;
					}
				}
				max_index_arr->at(i).at<uchar>(j,index) = (uchar)arg;
			}
	}

	Mat result = mat_vector_max(img_result);
	for (int i = 0; i < result.rows; ++i)
		for (int j = 0; j < result.cols; ++j)
		{
			if (result.at<float>(i, j) < 0)
				result.at<float>(i, j) = 0;
		}
	return result;
}

vector<Mat> generate_middle_stick_filter(int L)
{
	if (L % 2 == 0 || L < 3)
		assert("L should be an odd number and bigger than 3");

	int filter_num = (L - 1) / 2 + 1;

	vector<Mat> result;

	int r = (L - 1) / 2;
	for (int i = 0; i < filter_num; ++i)
	{
		Mat tmp = Mat::zeros(Size(L, L), CV_8S);
		tmp.at<schar>(r, r) = 1;
		for (int m = 0; m < r; ++m)
			for (int n = 0; n < r + 1; ++n)
			{
				if (n == int(m + i * (L - 2 * m) / (double)L + 0.5))
				{
					tmp.at<schar>(m, n) = 1;
					tmp.at<schar>(L - m - 1, L - n - 1) = 1;
				}
			}
		result.push_back(tmp);
	}

	for (int i = 0; i < r; ++i)
	{
		Mat tmp;
		flip(result[r - i - 1], tmp, 1);
		result.push_back(tmp);
	}
	for (int i = 1; i < L - 1; ++i)
	{
		result.push_back(result[L - 1 - i].t());
	}

	return result;
}

vector<Mat> generate_middle_minus_left_filter(vector<Mat> m_filters, int S)
{
	int L = m_filters.size() / 2 + 1;
	cout << L << endl;
	vector<Mat> result;
	for (int i = 0; i < m_filters.size(); ++i)
		result.push_back(m_filters[i].clone());
	for (int i = 0; i < result.size(); ++i)
		for (int m = 0; m < L; ++m)
			for (int n = 0; n < L; ++n)
			{
				if (result[i].at<schar>(m, n) == 1)
				{
					if (i < L)
					{
						if (n - S >= 0)
							result[i].at<schar>(m, n - S) = -1;
						else
							result[i].at<schar>(m, n) = 0;
					}
					else
					{
						if (m - S >= 0)
							result[i].at<schar>(m - S, n) = -1;
						else
							result[i].at<schar>(m, n) = 0;
					}
				}
			}
	return result;
}

vector<Mat> generate_middle_minus_right_filter(vector<Mat> m_filters, int S)
{
	int L = m_filters.size() / 2 + 1;
	cout << L << endl;
	vector<Mat> result;
	for (int i = 0; i < m_filters.size(); ++i)
		result.push_back(m_filters[i].clone());
	for (int i = 0; i < result.size(); ++i)
		for (int m = 0; m < L; ++m)
			for (int n = 0; n < L; ++n)
			{
				if (result[i].at<schar>(m, n) == 1)
				{
					if (i < L)
					{
						if (n + S < L)
							result[i].at<schar>(m, n + S) = -1;
						else
							result[i].at<schar>(m, n) = 0;
					}
					else
					{
						if (m + S < L)
							result[i].at<schar>(m + S, n) = -1;
						else
							result[i].at<schar>(m, n) = 0;
					}
				}
			}
	return result;
}

// three_view_result need transpose
vector<vector<Mat>> lung_fissure_enhance(vector<Mat> images, int l, int w, int k, bool is_left)
{
	vector<Mat> mid_kernels = generate_middle_stick_filter(l);
	vector<Mat> minus_left_kernels = generate_middle_minus_left_filter(mid_kernels, w);
	vector<Mat> minus_right_kernels = generate_middle_minus_right_filter(mid_kernels, w);

	vector<Mat> mid_kernels_end_view(mid_kernels.begin() + 5, mid_kernels.begin() + 15);
	vector<Mat> minus_left_kernels_end_view(minus_left_kernels.begin() + 5, minus_left_kernels.begin() + 15);
	vector<Mat> minus_right_kernels_end_view(minus_right_kernels.begin() + 5, minus_right_kernels.begin() + 15);
	vector<Mat> max_index_arr;
	Mat image = images[0];
	for (int i = 0; i < images.size(); ++i)
	{
		max_index_arr.push_back(Mat::zeros(image.size(), CV_8U));
	}
	
	vector<vector<Mat>> three_view_result;

	vector<Mat> front_view_result;
	for (int i = 0; i < image.rows; ++i)
	{
		Mat data = Mat::zeros(Size(image.cols, images.size()), CV_32F);
		for (int j = 0; j < images.size(); ++j)
			for (int k = 0; k < image.cols; ++k)
				data.at<float>(j, k) = images[j].at<float>(i, k);
		front_view_result.push_back(
			my_convolution_cascade(minus_left_kernels, minus_right_kernels, mid_kernels, data, i, false, k, nullptr)
		);
	}
	three_view_result.push_back(front_view_result);

	vector<Mat> end_view_result;
	for (int i = 0; i < image.cols; ++i)
	{
		Mat data = Mat::zeros(Size(image.rows, images.size()), CV_32F);
		for (int j = 0; j < images.size(); ++j)
			for (int k = 0; k < image.rows; ++k)
				data.at<float>(j, k) = images[j].at<float>(k, i);
		if (is_left)
		{
			end_view_result.push_back(
				my_convolution_cascade(minus_left_kernels, minus_right_kernels, mid_kernels, data, i, true, k, &max_index_arr)
			);
		}
		else
		{
			end_view_result.push_back(
				my_convolution_cascade(minus_left_kernels_end_view, minus_right_kernels_end_view, mid_kernels_end_view, data, i, false, k, nullptr)
			);
		}
	}
	three_view_result.push_back(end_view_result);
	
	if (is_left)
		three_view_result.push_back(max_index_arr);

	return three_view_result;
}

vector<vector<Mat>> filterTest(vector<Mat> data, bool is_left)
{
	int count = 0, i = 0;
	cudaGetDeviceCount(&count);
	for(; i < count; ++i)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	cout << count << " " << i << endl;
	assert(i < count);
	cudaSetDevice(i);
	
	return lung_fissure_enhance(data, 11, 2, 7, is_left);
}