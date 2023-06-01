#include "Layer.h"


Layer::Layer(int n_pre_node, int n_node)
{
    //가우시안 분포로 랜덤 생성
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(-1.0, 1.0);

    //weight 크기
    m_weight = Eigen::MatrixXd(n_pre_node, n_node);
    //bias 크기
    m_bias = Eigen::MatrixXd(1, n_node);

    //가중치와 bias 초기화

    for (int i = 0; i < n_node;i++) {
        double b_number = distribution(generator);
        m_bias(0, i) = b_number;
        for (int j = 0; j < n_pre_node;j++) {


            double number = distribution(generator);

            m_weight(j, i) = number;

        }
    }
    m_n_node = n_node;
    m_node = Eigen::MatrixXd(n_node, 1);

    cv::eigen2cv(m_weight, debug_matrix);

}

Eigen::MatrixXd Layer::calculate(Eigen::MatrixXd data)
{
    Eigen::MatrixXd result;
    result = data * m_weight + m_bias;
    m_node = result;

    return result;
}

Eigen::MatrixXd Layer::getWeight()
{
    return m_weight;
}

Eigen::MatrixXd Layer::getBias()
{
    return m_bias;
}

Eigen::MatrixXd Layer::getnode()
{
    return m_node;
}




void Layer::setWeight(Eigen::MatrixXd weight)
{
    m_weight = weight;
}

void Layer::setBias(Eigen::MatrixXd bias)
{
    m_bias = bias;
}