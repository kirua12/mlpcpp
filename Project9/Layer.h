#pragma once
#include <vector>
#include <random>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>



class Layer
{
	Eigen::MatrixXd m_weight;
	Eigen::MatrixXd m_bias;

	Eigen::MatrixXd m_node;
	int m_n_node;// nodeÀÇ °¹¼ö
public:
	Layer(int n_pre_node, int n_node);
	cv::Mat debug_matrix;

	Eigen::MatrixXd calculate(Eigen::MatrixXd data);
	Eigen::MatrixXd getWeight();
	Eigen::MatrixXd getBias();
	Eigen::MatrixXd getnode();



	void setWeight(Eigen::MatrixXd weight);
	void setBias(Eigen::MatrixXd bias);




};