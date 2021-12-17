#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>
#include <ctime>
#include "filter_test.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"
using namespace cv;
int nSendBuf=32*1024;//设置为32K


std::vector<Mat> encode_Result(Mat res){
        int mask = 255;
        int row = res.rows;
        int col = res.cols;
        vector<Mat> encoded_res;
        //分成4个uint8矩阵
        Mat endcoded_res_1(row,col,CV_8U);
        Mat endcoded_res_2(row,col,CV_8U);
        Mat endcoded_res_3(row,col,CV_8U);
        Mat endcoded_res_4(row,col,CV_8U);

        for(size_t p = 0 ; p < row ; p++){
                for(size_t q = 0 ; q < col; q++){
                    float a = res.at<float>(p,q);
//                    cout <<"a["<<p<<"]["<<q<<"]: "<<a<<endl;
                    int* pIVal = (int*)&a;
//                     cout <<"a["<<p<<"]["<<q<<"]: "<<hex<<*pIVal<<endl;
                    int a_0 = (uchar) *pIVal;


                    a_0 &= mask;
                    int a_1 =  (*pIVal)>>24;
                    a_1 &= mask;
                    int a_2 =  (*pIVal)>>16;
                    a_2 &= mask;
                    int a_3 = (*pIVal)>>8;
                    a_3 &= mask;
//                    cout <<"a["<<p<<"]["<<q<<"]2: "<<hex<<a_2<<endl;
//                    cout <<"a["<<p<<"]["<<q<<"]3: "<<hex<<a_3<<endl;

                    endcoded_res_1.at<uchar>(p,q) = a_1;
                    endcoded_res_2.at<uchar>(p,q) = a_2;
                    endcoded_res_3.at<uchar>(p,q) = a_3;
                    endcoded_res_4.at<uchar>(p,q) = a_0;

                }
            }
            encoded_res.push_back(endcoded_res_1);
            encoded_res.push_back(endcoded_res_2);
            encoded_res.push_back(endcoded_res_3);
            encoded_res.push_back(endcoded_res_4);

            return encoded_res;

}

void send_encode_Result(std::vector<Mat> encode_res,int server_sockfd,struct sockaddr_in remote_addr ,socklen_t sin_size,std::vector<int> quality){
    int len = 0;
    char encodeImg[65535];
    std::vector<uchar> data_encode;
    for (int x = 0;x <encode_res.size();x++){
    imencode(".png", encode_res[x], data_encode, quality);
    int nSize = data_encode.size();
    //cout<<"nSize:  "<<nSize<<endl;
    for (int i = 0; i < nSize; i++)
        {
        encodeImg[i] = data_encode[i];
        }

    if (len = sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
        {
            perror("send error");
//            return 1;
        }
//    cout<<"len:"<<len<<endl;
    memset(&encodeImg, 0, sizeof(encodeImg));
    }


}



int main()
{

    std::cout<<"Initialized"<<endl;
	int server_sockfd;
	std::vector<int> lens;
	int len=0;
	struct sockaddr_in my_addr;   //服务器网络地址结构体
	struct sockaddr_in remote_addr; //客户端网络地址结构体
	socklen_t sin_size;

	memset(&my_addr, 0, sizeof(my_addr)); //数据初始化--清零
	my_addr.sin_family = AF_INET; //设置为IP通信
	my_addr.sin_addr.s_addr = INADDR_ANY;//服务器IP地址--允许连接到所有本地地址上
	my_addr.sin_port = htons(8001); //服务器端口号



	/*创建服务器端套接字--IPv4协议，面向无连接通信，UDP协议*/
	if ((server_sockfd = socket(PF_INET, SOCK_DGRAM, 0)) < 0)
	{
		perror("socket error");
		return 1;
	}


	/*将套接字绑定到服务器的网络地址上*/
	if (bind(server_sockfd, (struct sockaddr*)&my_addr, sizeof(struct sockaddr)) < 0)
	{
		perror("bind error");
		return 1;
	}
	sin_size = sizeof(struct sockaddr_in);
	//printf("waiting for a packet...\n");

    setsockopt(server_sockfd,SOL_SOCKET,SO_SNDBUF,(const char*)&nSendBuf,sizeof(int));

	Mat image;
	Mat retimg;
	//char buf[65536] = {0};
	char littlebuf[1024] = { 0 };
	std::vector<Mat> imgstore_r;
	std::vector<Mat> imgstore_l;

	int times = 0;




	while (true)
	{   printf("waiting for a packet...\n");
//	    if(times != 0){
//            for(int i = 0;i<512;i++){
//            uchar buf[65536] = {0};
//            bigbuf[i] = buf;
//            }
//        }

		memset(&remote_addr, 0, sizeof(remote_addr));

			//接收图片张数
		if ((len = recvfrom(server_sockfd, littlebuf, sizeof(littlebuf), 0, (struct sockaddr*)&remote_addr, &sin_size)) < 0)
		{
			perror("recvfrom error");
			return 1;
		}
			printf("received packet from %s:\n", inet_ntoa(remote_addr.sin_addr));

			printf("numbers: %s\n", littlebuf);

			int numbers = atoi(littlebuf);


//
//		int numbers = *(int*) &littlebuf[4];
//
//		cout<<numbers<<endl;
//
//        return 0;

         std::vector<vector<uchar>>bigbuf;




		for (int num = 0; num < numbers; num++) {
            uchar buf[65536] = {0};

			if ((len = recvfrom(server_sockfd,buf, 65536, 0, (struct sockaddr*)&remote_addr, &sin_size)) < 0)
			{
				perror("recvfrom error");
				return 1;
			}
			printf("received packet from %s:\n", inet_ntoa(remote_addr.sin_addr));
			std::cout<<"num:"<<num<<endl;

			lens.push_back(len);
			std::cout<<"len:"<<len<<endl;
//			bigbuf[num][len] = '\0';

            vector<uchar> encodeItem ;	//用来存储单个图像字符串
			int pos = 0;
			while (pos < len)
			{
				encodeItem.push_back(buf[pos++]);//存入vector
			}

			bigbuf.push_back(encodeItem);

		}
		int m = 0;
		while (m < numbers) {


            vector<Mat> imgpool;

            for (int k = 0; k <4 ;k++){
            //cout<<"m:"<<m<<endl;
			vector<uchar> decodeItem ;	//用来存储单个图像字符串
			int pos = 0;
			while (pos < lens[m])
			{
				decodeItem.push_back(bigbuf[m][pos++]);//存入vector
			}

			image = imdecode(decodeItem, CV_LOAD_IMAGE_GRAYSCALE);//图像解码
//	        cout<<image<<endl;
            imgpool.push_back(image);

            m++;                                       //处理下一张图

            }


            //合并成一个Mat
            int row = imgpool[0].rows;
            int col = imgpool[0].cols;
//            cout<<imgpool[0].elemSize()<<endl;
//            cout<<row<<""<<col<<endl;

            Mat decodedImage(row,col,CV_32S);


            for(size_t p = 0 ; p < row ; p++){
                for(size_t q = 0 ; q < col; q++){
                    int de_1 =imgpool[0].at<uchar>(p,q);
                    int de_2 =imgpool[1].at<uchar>(p,q);
                    int de_3 =imgpool[2].at<uchar>(p,q);
                    int de_4 =imgpool[3].at<uchar>(p,q);

//                    cout<<"de_1:  "<<de_1<<endl;
//                    cout<<"de_2:  "<<de_2<<endl;
//                    cout<<"de_3:  "<<de_3<<endl;
//                    cout<<"de_4:  "<<de_4<<endl;

                    int ret_1 = (de_1<<24) | (de_2<<16);

//                    cout<<"ret_1:  "<<ret_1<<endl;



                    int ret_2 = (de_3<<8) | de_4;

//                    cout<<"ret_2:  "<<ret_2<<endl;

                    int ret = ret_1 | ret_2;
//                    cout<<"ret:  "<<ret<<endl;

                    decodedImage.at<int>(p,q) = ret;


                }
            }

            //cout<<decodedImage<<endl;
            decodedImage.convertTo(decodedImage, CV_32F);
            imgpool.clear();

            //return 0;


            if(times == 0){
            
			imgstore_r.push_back(decodedImage);					//存储
			//m++;

			}
			else{
			imgstore_l.push_back(decodedImage);					//存储
			//m++;
			}
		}
		lens.clear();

//
//             cout<<"print image"<<endl;
//             ofstream outfile;
//             outfile.open("t1");
//
//             outfile<<format(imgstore_r[50], Formatter::FMT_NUMPY);
//
//             outfile.close();
//
//             return 0;


        char encodeImg[65535];
		std::vector<int> quality;

		quality.push_back(CV_IMWRITE_PNG_COMPRESSION);
		quality.push_back(9);// 压缩级别

           //test
//        Mat H(2, 2, CV_32F);
//           for(int i = 0; i < H.rows; i++)
//                for(int j = 0; j < H.cols; j++)
//                      H.at<float>(i,j)=1./(i+j+1);
//
//        cout<<H<<endl;
//        std::vector<Mat> encodeList =  encode_Result(H);
//        send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);


//        cout<<encodeList[0]<<endl;
//        cout<<encodeList[1]<<endl;
//        cout<<encodeList[2]<<endl;
//        cout<<encodeList[3]<<endl;
//        std::vector<uchar> data_encode;
//        for (int x = 0;x <4;x++){
//        imencode(".png", encodeList[x], data_encode, quality);
//        int nSize = data_encode.size();
//        for (int i = 0; i < nSize; i++)
//			{
//			encodeImg[i] = data_encode[i];
//			}
//
//        if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
//					{
//						perror("send error");
//						return 1;
//					}
//				memset(&encodeImg, 0, sizeof(encodeImg));
//        }

//        return 0;



		//右肺
		if (times == 0) {
		    cout<<"right_lung:"<<endl;
		    cout<<"start:"<<endl;

            clock_t start = clock();
			vector<vector<Mat>> ret = filterTest(imgstore_r, false);
            clock_t end = clock();
            double time = (double)(end-start)/ CLOCKS_PER_SEC;
            cout<<"end."<<endl;
            cout<<"time:"<<time<<endl;


			int len1 = ret.size();
			int len2 = ret[0].size();
			int len3 = ret[1].size();
            cout<<"len1:"<<len1<<endl;
            cout<<"len2:"<<len2<<endl;
            cout<<"len3:"<<len3<<endl;

//             cout<<"print front_view_right"<<endl;
//             ofstream outfile1;
//             outfile1.open("t1");
//
//            for (int j2 = 0; j2 < len2; j2++) {
//             outfile1<<format(ret[0][j2], Formatter::FMT_NUMPY);
//                }
//             outfile1.close();

//            cout<<"type:"<<ret[0][0].type()<<endl;

           // return 0;

			for (int i2 = 0; i2 < len2; i2++) {

			    std::vector<Mat> encodeList =  encode_Result(ret[0][i2]);
//			    cout<<"rows:"<<encodeList[0].rows<<endl;
//			    cout<<"cols:"<<encodeList[0].cols<<endl;
                send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);
//
//
//				std::vector<uchar> data_encode;
//				imencode(".png", ret[0][i2], data_encode, quality);
//				int nSize = data_encode.size();
//				for (int i = 0; i < nSize; i++)
//					{
//						encodeImg[i] = data_encode[i];
//					}
//				if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
//					{
//						perror("send error");
//						return 1;
//					}
//				memset(&encodeImg, 0, sizeof(encodeImg));
			}



			for (int i3 = 0; i3 < len3; i3++) {

			    std::vector<Mat> encodeList =  encode_Result(ret[1][i3]);


                send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);
//
//					std::vector<uchar> data_encode;
//					imencode(".jpg", ret[1][i3], data_encode, quality);
//					int nSize = data_encode.size();
//					for (int i = 0; i < nSize; i++)
//					{
//						encodeImg[i] = data_encode[i];
//					}
//					if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
//					{
//						perror("send error");
//						return 1;
//					}
//					memset(&encodeImg, 0, sizeof(encodeImg));
			}
            imgstore_r.clear();
			times++;
		}//左肺
		else {
		    cout<<"left_lung:"<<endl;
			vector<vector<Mat>> ret = filterTest(imgstore_l, true);
			int len1 = ret.size();
			int len2 = ret[0].size();
            int len3 = ret[1].size();
            cout<<"len1:"<<len1<<endl;
            cout<<"len2:"<<len2<<endl;
            cout<<"len3:"<<len3<<endl;



			for (int i2 = 0; i2 < len2; i2++) {

			    std::vector<Mat> encodeList =  encode_Result(ret[0][i2]);
                send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);
//				std::vector<uchar> data_encode;
//				imencode(".jpg", ret[0][i2], data_encode, quality);
//				int nSize = data_encode.size();
//				for (int i = 0; i < nSize; i++)
//					{
//						encodeImg[i] = data_encode[i];
//					}
//				if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
//					{
//						perror("send error");
//						return 1;
//					}
//				memset(&encodeImg, 0, sizeof(encodeImg));
			}



			for (int i3 = 0; i3 < len3; i3++) {

			     std::vector<Mat> encodeList =  encode_Result(ret[1][i3]);
                 send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);

//					std::vector<uchar> data_encode;
//					imencode(".jpg", ret[1][i3], data_encode, quality);
//					int nSize = data_encode.size();
//					for (int i = 0; i < nSize; i++)
//					{
//						encodeImg[i] = data_encode[i];
//					}
//					if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
//					{
//						perror("send error");
//						return 1;
//					}
//					memset(&encodeImg, 0, sizeof(encodeImg));
			}

			//特殊处理左肺的max_index_arr

			if (len1 == 3) {
				int len4 = ret[2].size();
				cout<<"len4:"<<len4<<endl;

                //print test max_index_array
//			    cout<<"print max_index_array"<<endl;
//                ofstream outfile2;
//                outfile2.open("max_index");
//                for (int j3 = 0; j3 < len4; j3++){
//                outfile2<<format(ret[2][j3], Formatter::FMT_NUMPY);
//                }
//                outfile2.close();



				for (int i3 = 0; i3 < len4; i3++)
				{
//				    std::vector<Mat> encodeList =  encode_Result(ret[2][i3]);
//                    send_encode_Result(encodeList,server_sockfd, remote_addr,sin_size,quality);

					std::vector<uchar> data_encode;
					imencode(".png", ret[2][i3], data_encode, quality);
					int nSize = data_encode.size();
					for (int i = 0; i < nSize; i++)
					{
						encodeImg[i] = data_encode[i];
					}
					if (sendto(server_sockfd, encodeImg, nSize, 0, (struct sockaddr*)&remote_addr, sin_size) < 0)
					{
						perror("send error");
						return 1;
					}
					memset(&encodeImg, 0, sizeof(encodeImg));
				}
			}
			imgstore_l.clear();
			times = 0;
		}


	}





	/*关闭套接字*/
	close(server_sockfd);

	return 0;
}
