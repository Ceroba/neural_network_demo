#pragma once
#include"layer.h"
#include<thread>
struct data {
	std::vector<double> input;
	std::vector<double> output;
};
using Layercost = double(*)(const std::vector<double>&, const std::vector<double>& );


double net_cost(const std::vector<std::vector<double>> &expect, const std::vector<std::vector<double>>& pred, Layercost layer_cost) {
	double c = 0;
	for (int i = 0; i < expect.size(); i++) {
		c += layer_cost(expect[i], pred[i]);
	}
	return c / expect.size();
}
struct GradBuff {
	std::vector<std::vector<double>> grads;
};

class network
{
public:
	std::vector<int> _layersizes;
	uint16_t num_threads;
	std::vector<std::vector<double>> activations;
	std::vector<layer> layers;
	Layercost layer_cost;
	double cost(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect);
	network(const std::vector<int> &layersizes, bool is_CE_SM = 0);
	~network();
	std::vector<double> propugate(std::vector<double> inputs);
	void render(/*vx::renderer& r*/);
	void learnOld(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate);
	void learn(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect,double learnrate);
	void train_batch(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate, int batch_size);
	void train_batchOld(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate, int batch_size);
	void apply_all_grad(double learnrate);
	void clear_all_grad();
	void update_all_grad(const std::vector<double>& input, const std::vector<double>& expect);
private:

};

inline double network::cost(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect)
{
	std::vector<std::vector<double>> pred;
	for (int i = 0; i < input.size(); i++) {
		pred.push_back(propugate(input[i]));
	}
	auto loss = net_cost(expect, pred, layer_cost);
	return loss;
}

network::network(const std::vector<int>& layersizes, bool is_CE_SM)
{
	_layersizes = layersizes;
	layers.resize(layersizes.size() - 1);
	for (int i = 0; i < layers.size(); i++) {
		layers[i] = layer(layersizes[i], layersizes[i + 1]);
	}
	layers[layers.size() - 1].is_output = true;
	if (is_CE_SM) {
		layer_cost = cst::cross_entropy;
		layers[layers.size() - 1].out_activation = act::act_type::softmax;
	}
	else {
		layer_cost = cst::MSE;
		layers[layers.size() - 1].out_activation = act::act_type::sigmoid;
	}
	num_threads = std::thread::hardware_concurrency();
}

network::~network()
{
}

inline std::vector<double> network::propugate(std::vector<double> inputs)
{
	std::vector<double> in = inputs;
	activations.clear();
	activations.emplace_back(in);
	for (auto &l : layers) {
		in = l.calculate(in);
		activations.emplace_back(in);
	}

	return in;
}

//inline void network::render( /*vx::renderer& r*/)
//{
//	system("cls");
//	for (auto l : layers)
//	{
//		for (int i = 0; i < l.n_nodes_out; i++) {
//			std::cout << "* ";
//		}
//		std::cout << "\n";
//	}
//}

inline void network::learnOld(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect,double learnrate)
{
	const double h = 0.00001;
	double original_cost = cost(input, expect);
	for (auto& l : layers) {
		for (int nodeout = 0; nodeout < l.n_nodes_out; nodeout++) {
			for (int nodein = 0; nodein < l.n_nodes_in; nodein++) {
				l.weights[nodeout * l.n_nodes_in + nodein] += h;
				double d_cost = cost(input, expect) - original_cost;
				l.weights[nodeout * l.n_nodes_in + nodein] -= h;
				l.gradiant_w[nodeout * l.n_nodes_in + nodein] = d_cost / h;
			}
		}
		for (int i = 0; i < l.n_nodes_out; i++) {
			l.biases[i] += h;
			double d_cost = cost(input, expect) - original_cost;
			l.biases[i] -= h;
			l.gradiant_b[i] = d_cost / h;
		}
	}
	apply_all_grad(learnrate);
}

inline void network::learn(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate)
{
	for (int i = 0; i < input.size(); i++)
		update_all_grad(input[i], expect[i]);
	apply_all_grad(learnrate / input.size());
	clear_all_grad();
}

inline void network::train_batch(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate, int batch_size)
{
	for (int start = 0; start < input.size(); start += batch_size) {
		int end = std::min(start + batch_size, (int)input.size());
		std::vector<std::vector<double>> in(input.begin() + start, input.begin() + end);
		std::vector<std::vector<double>> out(expect.begin() + start, expect.begin() + end);
		learn(in, out, learnrate);
		//std::cout <<start << "\n";

	}
	//system("cls");
	//static int epoch = 0;
	//std::cout << "cost n" << epoch++ << ": " << cost(input, expect) << "\n";
}

inline void network::apply_all_grad(double learnrate)
{
	for (auto &l : layers)
		l.apply_grad(learnrate);
}
inline void network::clear_all_grad()
{
	for (auto& i : layers)
		i.clear_grad();
}
void add_grads(std::vector<double> a, std::vector<double> b) {
	for (int i = 0; i < a.size(); i++)
		a[i] += b[i];
}
inline void network::update_all_grad(const std::vector<double>& input, const std::vector<double>& expect)
{
	propugate(input);
	layer &out_layer = layers[layers.size() - 1];
	std::vector<double> node_vals = out_layer.calculate_output_node_values(expect);
	//grads.grads[layers.size() - 1] += node_vals;
	//add_grads(grads.grads[layers.size() - 1] , node_vals);
	out_layer.update_grad(node_vals);
	for (int i = layers.size() - 2; i >= 0; i--) {

		layer& hiddenlayer = layers[i];
		node_vals = hiddenlayer.calculate_hidden_layer_node_values(layers[i + 1], node_vals);
		//grads.grads[i] += node_vals;
		//grads.grads[i] = node_vals;
		//add_grads(grads.grads[i], node_vals);

		hiddenlayer.update_grad(node_vals);
	}
}
inline void network::train_batchOld(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& expect, double learnrate, int batch_size) {
	for (int start = 0; start < input.size(); start += batch_size) {
		int end = std::min(start + batch_size, (int)input.size());
		std::vector<std::vector<double>> in(input.begin() + start, input.begin() + end);
		std::vector<std::vector<double>> out(expect.begin() + start, expect.begin() + end);
		learnOld(in, out, learnrate);
	}
}