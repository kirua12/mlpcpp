#pragma once
#include <string>
#include <Eigen/Core>
#include <vector>
#include "Layer.h"
#include "Dataset.h"

struct BackProData {
	std::vector<Eigen::MatrixXd> nodes;
	std::vector<Eigen::MatrixXd> weights;

	std::vector<Eigen::MatrixXd> biases;
	std::vector<Eigen::MatrixXd> weights_grad;
	std::vector<Eigen::MatrixXd> biases_grad;
	double gt = 0;
	Eigen::MatrixXd data;

};


class Network
{
	int m_n_hidden; //���緹�̾� ��
	int m_n_node; // ��� ��
	double m_lr;// learning rate
	std::string m_activate; //Ȱ��ȭ �Լ�
	std::vector <Layer> m_layers;
	Dataset m_data;


public:
	Network();
	double forward(Eigen::MatrixXd data, double gt, double* error);
	Eigen::MatrixXd relu(Eigen::MatrixXd data);
	double lossMSE(double y_p, double y_g);
	void backPropagation(Eigen::MatrixXd data, double gt);

	void findGrad(BackProData* b_data);
};
