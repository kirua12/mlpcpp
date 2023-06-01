#include "Network.h"
#include "Dataset.h"
#include <iostream>
#include <fstream>

template<typename T>
inline void MakeTxt(std::vector<T> const& v, std::string textname)
{

	std::string f_name = textname + ".txt";
	std::ofstream file(f_name);

	if (file.is_open()) {
		for (int i = 0; i < v.size(); i++) {

			file << v.at(i);
			file << "\n";


		}
	}


	file.close();

}


int main() {
	Network net;
	Dataset dataset;
	std::vector<double> x_train = dataset.getXTrain();
	std::vector<double> y_train = dataset.getYTrain();



	std::vector<double> x_val = dataset.getXVal();
	std::vector<double> y_val = dataset.getYVal();
	std::vector<double> loss;
	std::vector<double> v_loss;

	std::vector<double> y_p;
	std::vector<double> y_vp;

	MakeTxt(x_train, "xt");
	MakeTxt(y_train, "yt");
	MakeTxt(x_val, "xv");
	MakeTxt(y_val, "yv");

	for (int i = 0; i < 500; i++) {
		double error_sum = 0;
		for (int j = 0; j < x_train.size(); j++) {

			Eigen::MatrixXd data(1, 1);
			data(0, 0) = x_train.at(j);
			double gt_y = y_train.at(j);
			double error = 0;
			data(0, 0) = net.forward(data, gt_y, &error);
			y_p.push_back(data(0, 0));
			error_sum += error;
			net.backPropagation(data, gt_y);
		}
		error_sum = error_sum / x_train.size();
		std::cout << error_sum <<"   "<< i << std::endl;
		loss.push_back(error_sum);

		for (int j = 0; j < x_val.size(); j++) {

			Eigen::MatrixXd data(1, 1); 
			data(0, 0) = x_val.at(j);
			double gt_y = y_val.at(j);
			double error = 0;
			data(0, 0) = net.forward(data, gt_y, &error);
			y_vp.push_back(data(0, 0));
			error_sum += error;
		}

		error_sum = error_sum / x_val.size();
		v_loss.push_back(error_sum);

		MakeTxt(loss, "loss");
		MakeTxt(v_loss, "vloss");

		MakeTxt(y_p, "yp");
		MakeTxt(y_vp, "yvp");
		y_vp.clear();
		y_p.clear();
		loss.clear();
		v_loss.clear();



	}




	for (int j = 0; j < x_train.size(); j++) {
		double error;
		Eigen::MatrixXd data(1, 1);

		data(0, 0) = x_train.at(j);
		double gt_y = y_train.at(j);
		data(0, 0) = net.forward(data, gt_y, &error);
		y_p.push_back(data(0, 0));
	}


	for (int j = 0; j < x_val.size(); j++) {
		double error;
		Eigen::MatrixXd data(1, 1);

		data(0, 0) = x_val.at(j);
		double gt_y = y_val.at(j);
		data(0, 0) = net.forward(data, gt_y, &error);
		y_vp.push_back(data(0, 0));
	}

	MakeTxt(y_p, "yp");
	MakeTxt(y_vp, "yvp");



}