
#include "Network.h"

Network::Network()
{
	m_n_hidden = 4;
	m_n_node = 30;
	m_lr = 0.005;

	Layer l1(1, m_n_node);
	m_layers.push_back(l1);
	for (int i = 0; i < m_n_hidden - 1; i++) {
		Layer layer(m_n_node, m_n_node);
		m_layers.push_back(layer);
	}

	Layer le(m_n_node, 1);
	m_layers.push_back(le);

	int a = 1;
}

double Network::forward(Eigen::MatrixXd data, double gt, double* error)
{

	//모든 error의 총합
	for (int j = 0; j < m_layers.size();j++) {
		data = m_layers.at(j).calculate(data);


		if (j == m_layers.size() - 1) continue;// 마지막은 relu를 적용하디 않는다
		data = relu(data);




	}
	*error = lossMSE(data(0, 0), gt);

	return data(0, 0);
}

Eigen::MatrixXd Network::relu(Eigen::MatrixXd data)
{
	for (int i = 0; i < data.cols();i++) {

		data(0, i) = std::tanh(data(0, i));

	}
	return data;
}

double Network::lossMSE(double y_p, double y_g)
{
	return std::pow(y_p - y_g, 2) / 2;
}



void Network::backPropagation(Eigen::MatrixXd data, double gt)
{

	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::MatrixXd> biases;


	for (int j = 0; j < m_layers.size(); j++) {
		Eigen::MatrixXd weight = m_layers.at(j).getWeight();
		Eigen::MatrixXd bias = m_layers.at(j).getBias();
		weights.push_back(weight);
		biases.push_back(bias);
	}

	BackProData b_data;
	b_data.biases = biases;
	b_data.weights = weights;
	b_data.data = data;
	b_data.gt = gt;

	findGrad(&b_data);


	for (int j = 0; j < m_layers.size(); j++) {
		Eigen::MatrixXd weight = m_layers.at(j).getWeight();
		Eigen::MatrixXd bias = m_layers.at(j).getBias();
		Eigen::MatrixXd bias_g = b_data.biases_grad.at(j);
		Eigen::MatrixXd weight_g = b_data.weights_grad.at(j);
		bias_g = m_lr * bias_g;
		bias = bias - bias_g;
		weight_g = m_lr * weight_g;
		weight = weight - weight_g;
		m_layers.at(j).setWeight(weight);
		m_layers.at(j).setBias(bias);


	}

	weights.clear();
	biases.clear();
	b_data.biases.clear();
	b_data.weights.clear();
	b_data.weights_grad.clear();
	b_data.biases_grad.clear();
	

}

void Network::findGrad(BackProData* b_data)
{
	std::vector<Eigen::MatrixXd> weights = (*b_data).weights;
	std::vector<Eigen::MatrixXd> biases = (*b_data).biases;

	double gt = (*b_data).gt;
	Eigen::MatrixXd data = (*b_data).data;


	double h = 0.001;


	//weight에 기울기를 구하는 과정
	for (int i = 0; i < weights.size(); i++) {
		Eigen::MatrixXd weight = weights.at(i);
		Eigen::MatrixXd weight_grad = Eigen::MatrixXd(weight.rows(), weight.cols());
		Eigen::MatrixXd temp_w = weight;

		//미분과정
		for (int j = 0; j < weight.rows(); j++) {
			for (int k = 0; k < weight.cols(); k++) {

				//f1
				weight(j, k) = temp_w(j, k) + h;
				m_layers.at(i).setWeight(weight);
				double error1;
				forward(data, gt, &error1);

				//f2
				weight(j, k) = temp_w(j, k) - h;
				m_layers.at(i).setWeight(weight);
				double error2;
				forward(data, gt, &error2);
				//미분
				double grad = (error1 - error2) / (2 * h);
				weight_grad(j, k) = grad;
			}
		}

		(*b_data).weights_grad.push_back(weight_grad);
		m_layers.at(i).setWeight(temp_w);
	}



	for (int i = 0; i < biases.size(); i++) {
		Eigen::MatrixXd bias = biases.at(i);
		Eigen::MatrixXd bias_grad = Eigen::MatrixXd(bias.rows(), bias.cols());
		Eigen::MatrixXd temp_b = bias;

		//미분과정
		for (int j = 0; j < bias.rows(); j++) {
			for (int k = 0; k < bias.cols(); k++) {

				//f1
				bias(j, k) = temp_b(j, k) + h;
				m_layers.at(i).setBias(bias);
				double error1;
				forward(data, gt, &error1);

				//f2
				bias(j, k) = temp_b(j, k) - h;
				m_layers.at(i).setBias(bias);
				double error2;
				forward(data, gt, &error2);
				//미분
				double grad = (error1 - error2) / (2 * h);
				bias_grad(j, k) = grad;
			}
		}

		(*b_data).biases_grad.push_back(bias_grad);
		m_layers.at(i).setBias(temp_b);
	}








}