#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iterator> 
#include <algorithm>  
using namespace cv;
using namespace std;

int num = 10; //采集的数据对数
//相机到标定板的位姿：{x,y,z,w,x,y,z}
Mat_<double> CalPose = (cv::Mat_<double>(num, 7) <<
	-0.021582037, 0.010825566016, 0.1273279786, -0.22801148538, 0.9582988210737, -0.1680881472, 0.0376896116,
	0.0121286856, 0.0099868699, 0.1394957602, -0.211499174269, 0.7753145282, -0.5924676707, -0.05601375469,
	0.0164144877, -0.01038888748, 0.14443306624, 0.3394615017, -0.63986956044, 0.68891345143, 0.0271177217,
	-0.0002430716, -0.018399436, 0.1439810693264, 0.4673437139, -0.52681168939, 0.70817224155, -0.0505111201151,
	0.0080916211, -0.0129763754, 0.14114294946, 0.1396090045, -0.41426447483, 0.861733517, -0.25750653529,
	0.015907838, -0.0176754817, 0.1371374279, 0.1711303204289, -0.128104683117, 0.8603046619314, -0.4627953027,
	0.0143387466, -0.01958738081, 0.12833982706, 0.18848043807, 0.4530692354, 0.83411153347, -0.251915347,
	0.038024716, -0.030567344, 0.221145942807, -0.26623783212, 0.59044434879, 0.7487989959, 0.14068742434,
	-0.00228397897, -0.02677887678, 0.1820683628, 0.0876127144829, 0.546350691737, 0.8168659212, -0.1629570516,
	0.02192275039, -0.029831523075, 0.15898549556, -0.06875009809012, 0.486094971489, 0.7745623121569, -0.3987960973872);
//基座标系到机械臂末端的位姿：{x,y,z,Rx,Ry,Rz}
Mat_<double> ToolPose = (cv::Mat_<double>(num, 6) <<
	0.05587, -0.41588, -0.04057, 2.235, 1.578, -0.424,
	0.07428,-0.46297,-0.03136, 2.841, 0.388, -0.587,
	0.06785,-0.46295,-0.03134, 2.609, -0.092, -0.739,
	0.06787,-0.46296,-0.03135, 2.256, -0.332, -0.781,
	0.00625,-0.46298,0.00306, 2.434, -0.844, 0.213,
	0.00627,-0.42855,-0.01855, 1.735, -1.264, 0.518,
	0.00626,-0.46657,-0.01055, 0.728, -2.396, 0.125,
	-0.1195,-0.47881,0.04138, 0.458, -3.702, 0.394,
	0.00312,-0.44054,0.03630, 0.561, -2.732, 0.167,
	0.00314,-0.40977,-0.01010, 0.575, -2.440, 0.906);
//R和T转RT矩阵
Mat R_T2RT(Mat& R, Mat& T)
{
	Mat RT;
	Mat_<double> R1 = (cv::Mat_<double>(4, 3) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
		0.0, 0.0, 0.0);
	cv::Mat_<double> T1 = (cv::Mat_<double>(4, 1) << T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0), 1.0);

	cv::hconcat(R1, T1, RT);//C=A+B左右拼接
	return RT;
}

//RT转R和T矩阵
void RT2R_T(Mat& RT, Mat& R, Mat& T)
{
	cv::Rect R_rect(0, 0, 3, 3);
	cv::Rect T_rect(3, 0, 1, 3);
	R = RT(R_rect);
	T = RT(T_rect);
}

//判断是否为旋转矩阵
bool isRotationMatrix(const cv::Mat& R)
{
	cv::Mat tmp33 = R({ 0,0,3,3 });
	cv::Mat shouldBeIdentity;

	shouldBeIdentity = tmp33.t() * tmp33;

	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

	return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

// @brief 四元数转旋转矩阵

cv::Mat quaternionToRotatedMatrix(const cv::Vec4d& q)
{
	double w = q[0], x = q[1], y = q[2], z = q[3];

	double x2 = x * x, y2 = y * y, z2 = z * z;
	double xy = x * y, xz = x * z, yz = y * z;
	double wx = w * x, wy = w * y, wz = w * z;

	cv::Matx33d res{
		1 - 2 * (y2 + z2),	2 * (xy - wz),		2 * (xz + wy),
		2 * (xy + wz),		1 - 2 * (x2 + z2),	2 * (yz - wx),
		2 * (xz - wy),		2 * (yz + wx),		1 - 2 * (x2 + y2),
	};
	return cv::Mat(res);
}

// @brief 四元数 -> 4*4 的Rt
cv::Mat attitudeVectorToMatrix(cv::Mat m)
{
	CV_Assert(m.total() == 6 || m.total() == 7);
	if (m.cols == 1)
		m = m.t();
	cv::Mat tmp = cv::Mat::eye(4, 4, CV_64FC1);
	//如果使用四元数转换成旋转矩阵则读取m矩阵的第四个成员，读4个数据
	cv::Vec4d quaternionVec = m({ 3, 0, 4, 1 });
	quaternionToRotatedMatrix(quaternionVec).copyTo(tmp({ 0, 0, 3, 3 }));//四元数2旋转矩阵
	// cout << norm(quaternionVec) << endl; 
	// cout << "----------------------------" << endl; 
	// cout << quaternionVec << endl; 
	tmp({ 3, 0, 1, 3 }) = m({ 0, 0, 3, 1 }).t();
	//std::swap(m,tmp);
	// cout << "###########################" << endl;
	// cout << tmp << endl;
	return tmp;
}

//旋转矢量 -> 4*4 的Rt
cv::Mat rotationVectorToMatrix(cv::Mat m)
{
	CV_Assert(m.total() == 6 || m.total() == 7);
	if (m.cols == 1)
		m = m.t();
	cv::Mat tmp = cv::Mat::eye(4, 4, CV_64FC1);
	cv:: Mat rotVec ;
	rotVec = m({ 3, 0, 3, 1 });	
	cv::Rodrigues(rotVec, tmp({ 0, 0, 3, 3 }));
	tmp({ 3, 0, 1, 3 }) = m({ 0, 0, 3, 1 }).t();
	return tmp;
}

int main()
{
	//定义手眼标定矩阵
	std::vector<Mat> R_gripper2base;
	std::vector<Mat> t_gripper2base;
	std::vector<Mat> R_target2cam;
	std::vector<Mat> t_target2cam;
	Mat R_cam2gripper = (Mat_<double>(3, 3));
	Mat t_cam2gripper = (Mat_<double>(3, 1));

	vector<Mat> images;
	size_t num_images = num;

	// 读取末端，标定板的姿态矩阵 4*4
	std::vector<cv::Mat> vecHb, vecHc;
	cv::Mat Hcb;//定义相机camera到末端grab的位姿矩阵
	Mat tempR, tempT;

	for (size_t i = 0; i < num_images; i++)//计算标定板位姿
	{
		cv::Mat tmp = attitudeVectorToMatrix(CalPose.row(i)); //转移向量转旋转矩阵
		// tmp = tmp.inv();
		vecHc.push_back(tmp);//camera位姿齐次矩阵
		RT2R_T(tmp, tempR, tempT);
		// cout << "##############################" << endl;
		// cout << tempR << endl << tempT << endl;
		R_target2cam.push_back(tempR);
		t_target2cam.push_back(tempT);
	}
	//cout << R_target2cam << endl;
	for (size_t i = 0; i < num_images; i++)//计算机械臂位姿
	{
		cv::Mat tmp = rotationVectorToMatrix(ToolPose.row(i)); //机械臂位姿为旋转矢量-旋转矩阵
		cout << "############" << i <<"##############" << endl;
		cout << tmp << endl;
		// tmp = tmp.inv();
		vecHb.push_back(tmp);
		RT2R_T(tmp, tempR, tempT);

		R_gripper2base.push_back(tempR);
		t_gripper2base.push_back(tempT);

	}

	//手眼标定
	calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper, t_cam2gripper, CALIB_HAND_EYE_TSAI);
	Hcb = R_T2RT(R_cam2gripper, t_cam2gripper);//矩阵合并

	std::cout << "Hcb 矩阵为： " << std::endl;
	std::cout << Hcb << std::endl;
	cout << "是否为旋转矩阵：" << isRotationMatrix(Hcb) << std::endl << std::endl;//判断是否为旋转矩阵

	//使用各组数据进行对比验证
	for (size_t i = 0; i < num_images; i++)
	{
		cout << "第"<< i <<"组手眼数据-base2target:" << endl;
		cout << vecHb[i] * Hcb * vecHc[i] << endl;
		//             g2b * c2g * t2c
	}
	//.inv()

	cout << "标定板在相机中的位姿：" << endl;
	cout << vecHc[3] << endl;
	cout << "手眼系统反演的位姿为：" << endl;
	//用手眼系统预测第一组数据中标定板相对相机的位姿，是否与vecHc[1]相同
	cout << Hcb.inv() * vecHb[3].inv() * vecHb[2] * Hcb * vecHc[2] << endl << endl;

	cout << "----手眼系统测试----" << endl;
	cout << "机械臂下标定板XYZ为" << endl;
	for (int i = 0; i < vecHc.size(); ++i)
	{
		cv::Mat cheesePos{ 0.0,0.0,0.0,1.0 };//4*1矩阵，单独求机械臂下，标定板的xyz
		cv::Mat worldPos = vecHb[i] * Hcb * vecHc[i] * cheesePos;
		cout << i << ": " << worldPos.t() << endl;
	}
	getchar();
}