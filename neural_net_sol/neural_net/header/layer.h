#pragma once
#include<array>
#include<vector>
#include<cmath>
#include<iostream>
#include <random>
#include"Activation.h"
#include "Cost.h"
using ActivationFunc = double(*)(double);
using OutActivationFunc = double(*)(std::vector<double>);

double _layer_cost(const std::vector<double>& expect, const std::vector<double>& pred) {
	double c = 0;
	for (int i = 0; i < expect.size(); i++) {
		double diff = expect[i] - pred[i];
		c += diff * diff;
	}
	return c / expect.size();
}
double d_node_cost(double expect, double pred) {
	return pred - expect;
}
//inline double sigmoid(double x) {
//	return 1.0 / (1.0 + std::exp(-x));
//}
//double d_sigmoid(double x) {
//	double act = sigmoid(x);
//	return act * (1 - act);
//}
//class IActivation {
//public:
//	virtual double activate(double x){};
//	virtual double D_activation(double x){};
//};
//class Activation {
//	enum Activation_type
//	{
//		Sigmoid = 0,
//		TanH = 1,
//		ReLu = 2,
//		SoftMax = 3
//	};
//	class Sigmoid : IActivation {
//		double activate(double x)override {
//			return 1.0 / (1.0 + std::exp(-x));
//		}
//		double D_activation(double x)override {
//			double act = sigmoid(x);
//			return act * (1 - act);
//		}
//	};
//	IActivation get_activation_by_type(Activation_type act) {
//		switch (act)
//		{
//		case Activation::Sigmoid:
//			return Sigmoid;
//			break;
//		case Activation::TanH:
//			break;
//		case Activation::ReLu:
//			break;
//		case Activation::SoftMax:
//			break;
//		default:
//			break;
//		}
//	}
//};
class layer
{
public:
	//learn date
	struct layerleadata
	{
		std::vector<double> input;
		std::vector<double> weighted_input;
		std::vector<double> activations;
	} learn_data;
	int n_nodes_in, n_nodes_out;
	//x-for innode y for outnode
	std::vector<double> weights;
	std::vector<double> biases;
	std::vector<double> gradiant_w;
	std::vector<double> gradiant_b;
	act::act_type out_activation;
	ActivationFunc activation;
	ActivationFunc D_activation;
	bool is_output = false;

	layer(int nodes_in, int nodes_out);
	layer() {};
	std::vector<double> calculate(const std::vector<double>& wiaghted_input);
	void apply_grad(double learnrate);
	void update_grad(const std::vector<double>& node_vals);
	//backprop
	std::vector<double> calculate_output_node_values(const std::vector<double>& expect);
	std::vector<double> calculate_hidden_layer_node_values(const layer &old,const std::vector<double>& oldnode_vals);
	void clear_grad();
};

layer::layer(int nodes_in, int nodes_out)
{
	n_nodes_in = nodes_in;
	n_nodes_out = nodes_out;
	std::uniform_real_distribution<double> dist(-1, 1);
	std::random_device rnd;
	weights.resize(n_nodes_in * n_nodes_out);
	gradiant_w.resize(n_nodes_in * n_nodes_out);
	for (auto &i : weights) {
		i = dist(rnd) / sqrt(n_nodes_in);
	}
	biases.resize(n_nodes_out);
	gradiant_b.resize(n_nodes_out);
	
	for (auto& i : biases) {
		i = dist(rnd);
	}
	learn_data.weighted_input.resize(n_nodes_out);
	learn_data.activations.resize(n_nodes_out);
	activation = act::Sigmoid;
	D_activation = act::D_Sigmoid;
}

inline std::vector<double> layer::calculate(const std::vector<double>& wiaghted_input)
{	
	learn_data.input = wiaghted_input;
	double wh_in = 0;
	for (int node_out = 0; node_out < n_nodes_out; node_out++) {
		wh_in = biases[node_out];
		for (int node_in = 0; node_in < n_nodes_in; node_in++) {
			double w = weights[node_out * n_nodes_in + node_in];
			wh_in += wiaghted_input[node_in] * w;
		}
		learn_data.weighted_input[node_out] = wh_in;
	}
	if (is_output && (out_activation == act::act_type::softmax)) {
		learn_data.activations = act::Softmax(learn_data.weighted_input);
	}
	else
		for (int i = 0; i < learn_data.activations.size(); i++) {
			learn_data.activations[i] = activation(learn_data.weighted_input[i]);
		}
	return learn_data.activations;

}

inline void layer::apply_grad(double learnrate)
{
	for (int nodeout = 0; nodeout < n_nodes_out; nodeout++) {
		biases[nodeout] -= gradiant_b[nodeout] * learnrate;
		for (int nodein = 0; nodein < n_nodes_in; nodein++) {
			int idx = nodeout * n_nodes_in + nodein;
			weights[idx] -= gradiant_w[idx] * learnrate;
		}
	}
}

inline void layer::update_grad(const std::vector<double>& node_vals)
{
	// For each output neuron
	for (int nodeout = 0; nodeout < n_nodes_out; nodeout++) {

		// Bias gradient
		gradiant_b[nodeout] += node_vals[nodeout];

		// Weight gradients
		for (int nodein = 0; nodein < n_nodes_in; nodein++) {
			int idx = nodeout * n_nodes_in + nodein;
			gradiant_w[idx] += learn_data.input[nodein] * node_vals[nodeout];
		}
	}
}

inline std::vector<double> layer::calculate_output_node_values(const std::vector<double>& expect)
{
	std::vector<double> node_vals(expect.size());
	if (out_activation == act::act_type::softmax)
		for (int i = 0; i < expect.size(); i++)
			node_vals[i] = learn_data.activations[i] - expect[i];
	else
		for (int i = 0; i < node_vals.size(); i++) {
			double dC_da = d_node_cost(expect[i], learn_data.activations[i]);
			double da_dz = D_activation(learn_data.weighted_input[i]);
			node_vals[i] = dC_da * da_dz; // δ = dC/da * da/dz
		}
	return node_vals;
}

inline std::vector<double> layer::calculate_hidden_layer_node_values(const layer& next_layer,
	const std::vector<double>& next_node_vals)
{
	std::vector<double> node_vals(n_nodes_out, 0.0);
	for (int i = 0; i < n_nodes_out; i++) {
		double sum = 0.0;
		for (int j = 0; j < next_layer.n_nodes_out; j++) {
			int idx = j * n_nodes_out + i; // weight from i→j
			sum += next_layer.weights[idx] * next_node_vals[j];
		}
		double da_dz = D_activation(learn_data.weighted_input[i]);
		node_vals[i] = sum * da_dz; // δ = Σ(next_weight * next_δ) * σ'(z)
	}
	return node_vals;
}


inline void layer::clear_grad()
{
	for (auto& i : gradiant_b)
		i = 0;
	for (auto& i : gradiant_w)
		i = 0;
}

//inline void layer::apply_grad(double learnrate)
//{
//	for (int nodeout = 0; nodeout < n_nodes_out; nodeout++) {
//		biases[nodeout] -= gradiant_b[nodeout] * learnrate;
//		for (int nodein = 0; nodein < n_nodes_in; nodein++) {
//			weights[nodeout * n_nodes_in + nodein] -= gradiant_w[nodeout * n_nodes_in + nodein] * learnrate;
//		}
//	}
//}
//
//inline void layer::update_grad(const std::vector<double>& node_vals)
//{
//	for (int nodeout = 0; nodeout < n_nodes_out; nodeout++) {
//
//		for (int nodein = 0; nodein < n_nodes_in; nodein++) {
//			double x = learn_data.input[nodein] * learn_data.activations[nodeout];
//			gradiant_w[nodeout * n_nodes_in + nodein] += x;
//		}
//		double y = 1 * node_vals[nodeout];
//		gradiant_b[nodeout] += y;
//	}
//}
//
//inline std::vector<double> layer::calculate_output_node_values(const std::vector<double>& expect)
//{
//	std::vector<double> node_vals(expect.size());
//	for (int i = 0; i < node_vals.size(); i++) {
//		double D_cost = d_node_cost(expect[i], learn_data.activations[i]);
//		double D_act = D_sigmoid(learn_data.weighted_input[i]);
//		node_vals[i] = D_act * D_cost;
//	}
//	return node_vals;
//}
//
//inline std::vector<double> layer::calculate_hidden_layer_node_values(const layer& old, const std::vector<double>& oldnode_vals)
//{
//	std::vector<double> newnode_vals(n_nodes_out);
//	for (int newnode = 0; newnode < newnode_vals.size(); newnode++) {
//		double node_val = 0;
//		for (int oldnode = 0; oldnode < oldnode_vals.size(); oldnode++) {
//			double D_weightedin = old.weights[newnode * oldnode_vals.size() + oldnode];
//			node_val += D_weightedin * oldnode_vals[oldnode];
//		}
//		node_val *= D_sigmoid(learn_data.weighted_input[newnode]);
//		newnode_vals[newnode] = node_val;
//	}
//	return newnode_vals;
//}