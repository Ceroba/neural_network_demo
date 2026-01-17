#include <cmath>
#include <algorithm>
#include <SFML/Graphics.hpp>
#include "..\header\network.h"
#include <SFML/Graphics.hpp>
#include <cmath>
#include "..\header\mnist_reader.hpp"
sf::Color activationColor(double value) {
	// interpolate between blue (0) -> white (0.5) -> red (1)
	sf::Uint8 r = static_cast<sf::Uint8>(255 * value);
	sf::Uint8 g = static_cast<sf::Uint8>(255 * value);
	sf::Uint8 b = static_cast<sf::Uint8>(255 * value);
	return sf::Color(r, g, b);
}

class Button {
public:
	Button(
		const sf::Vector2f& position,
		const sf::Vector2f& size,
		const std::string& textStr,
		const sf::Font& font,
		unsigned int charSize = 24
	) {
		shape.setPosition(position);
		shape.setSize(size);
		shape.setFillColor(sf::Color(80, 80, 80));

		text.setFont(font);
		text.setString(textStr);
		text.setCharacterSize(charSize);
		text.setFillColor(sf::Color::White);

		// Center the text inside the button
		sf::FloatRect textBounds = text.getLocalBounds();
		text.setOrigin(textBounds.left + textBounds.width / 2.f,
			textBounds.top + textBounds.height / 2.f);
		text.setPosition(position.x + size.x / 2.f, position.y + size.y / 2.f);

		colorNormal = sf::Color(80, 80, 80);
		colorHover = sf::Color(120, 120, 120);
		colorClicked = sf::Color(180, 180, 180);
	}

	void draw(sf::RenderWindow& window) {
		window.draw(shape);
		window.draw(text);
	}

	void update(const sf::RenderWindow& window) {
		if (isClicked(window)) {
			shape.setFillColor(colorClicked);
			wasPressed = true;
		}
		else if (isHovered(window)) {
			shape.setFillColor(colorHover);
			if (wasPressed && !sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
				wasPressed = false;
				clickedThisFrame = true;
			}
		}
		else {
			shape.setFillColor(colorNormal);
			wasPressed = false;
		}
	}

	bool isHovered(const sf::RenderWindow& window) const {
		auto mousePos = sf::Mouse::getPosition(window);
		return shape.getGlobalBounds().contains((float)mousePos.x, (float)mousePos.y);
	}

	bool isClicked(const sf::RenderWindow& window) const {
		return isHovered(window) && sf::Mouse::isButtonPressed(sf::Mouse::Left);
	}

	bool wasClicked() {
		if (clickedThisFrame) {
			clickedThisFrame = false;
			return true;
		}
		return false;
	}

	sf::RectangleShape shape;
	sf::Text text;
	sf::Color colorNormal, colorHover, colorClicked;
	bool wasPressed = false;
	bool clickedThisFrame = false;
private:
};
float clamp(float x, float min, float max) {
	return std::min(std::max(x, min), max);
}
class Slider {
public:
	Slider(float x, float y, float width, float min, float max)
		: m_min(min), m_max(max), m_value(min), m_width(width), m_dragging(false)
	{
		m_bar.setSize({ width, 4 });
		m_bar.setFillColor(sf::Color(180, 180, 180));
		m_bar.setPosition(x, y);

		m_knob.setRadius(8);
		m_knob.setOrigin(8, 8);
		m_knob.setFillColor(sf::Color::White);
		m_knob.setPosition(x, y + 2);
	}

	void handleEvent(const sf::Event& event, const sf::RenderWindow& window)
	{
		if (event.type == sf::Event::MouseButtonPressed &&
			event.mouseButton.button == sf::Mouse::Left)
		{
			sf::Vector2f mouse = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			if (m_knob.getGlobalBounds().contains(mouse))
				m_dragging = true;
		}
		else if (event.type == sf::Event::MouseButtonReleased)
			m_dragging = false;

		if (m_dragging && event.type == sf::Event::MouseMoved)
		{
			sf::Vector2f mouse = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			float left = m_bar.getPosition().x;
			float right = left + m_width;
			float pos = clamp(mouse.x, left, right);

			float t = (pos - left) / m_width;
			m_value = m_min + t * (m_max - m_min);

			m_knob.setPosition(pos, m_bar.getPosition().y + 2);
		}
	}

	void draw(sf::RenderWindow& window, sf::Font& font)
	{
		window.draw(m_bar);
		window.draw(m_knob);

		sf::Text label;
		label.setFont(font);
		label.setCharacterSize(16);
		label.setFillColor(sf::Color::White);

		char buffer[32];
		snprintf(buffer, sizeof(buffer), "%.2f", m_value);
		label.setString(buffer);

		float labelX = m_knob.getPosition().x - 10;
		float labelY = m_knob.getPosition().y - 30;
		label.setPosition(labelX, labelY);

		window.draw(label);
	}

	float getValue() const { return m_value; }
	void setValue(float v)
	{
		m_value = clamp(v, m_min, m_max);
		float t = (m_value - m_min) / (m_max - m_min);
		float pos = m_bar.getPosition().x + t * m_width;
		m_knob.setPosition(pos, m_bar.getPosition().y + 2);
	}

private:
	sf::RectangleShape m_bar;
	sf::CircleShape m_knob;
	float m_min, m_max;
	float m_value;
	float m_width;
	bool m_dragging;
};

int main() {
	std::vector<int> layersizes = { 2, 5, 5, 2};
	network net(layersizes, false);
	std::vector<std::vector<double>> inputs; //I hate using vector<vector<>>
	std::vector<std::vector<double>> outputs;

	sf::RenderWindow window(sf::VideoMode::getDesktopMode(), "Neural Network Visualizer", sf::Style::Default);
	window.setFramerateLimit(1000);

	sf::Font font;
	font.loadFromFile("m6x11plus.ttf");
	// layout parameters
	float xSpacing = 200.0f;
	float ySpacing = 80.0f; 
	float radius = 20.0f;   


	
	//sp.setScale(sf::Vector2f(10, 10));
	Button button({ 1000, 250 }, { 200, 80 }, "learn", font);
	Button button2({ 1000, 500 }, { 200, 80 }, "render", font);
	Slider slider(200, 300, 400, 0.0f, 1.0f);
	slider.setValue(0.5f);
	bool learn = false;
	double loss = 0;
	bool render = false;
	// Path where the MNIST .ubyte files are stored
	//std::string dataset_dir = "C:\\Users\\21355\\Desktop\\c++ files\\sfml environment\\neural_net_sol\\neural_net\\dataset";
	//std::ifstream test(dataset_dir + "\\train-images-idx3-ubyte", std::ios::binary);
	//std::cout << "File exists: " << test.good() << std::endl;
	// Load the dataset (images + labels)
	//auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dataset_dir)
	// Normalize pixel values to [0, 1]
	sf::Image img;
	float img_w = 256, img_h = 256;
	img.create(img_w, img_h);
	for (int i = 0; i < (int)img_h; i++) {
		for (int j = 0; j < (int)img_w; j++) {
			//sf::Uint8 pix = dataset.training_images[2][i * 28 + j];
			img.setPixel(j, i, sf::Color(0, 0, 0, 255));
		}
	}
	sf::Texture tex;
	tex.loadFromImage(img);
	sf::Sprite sp(tex);
	//sp.setScale(sf::Vector2f(10, 10));
	sp.setPosition(sf::Vector2f(600, 10));
	//std::cout << "Training images: " << dataset.training_images.size() << "\n";
	//std::cout << "First label: " << (int)dataset.training_labels[0] << "\n";
	//outputs.resize(dataset.training_labels.size());
	//for (int i = 0; i < dataset.training_labels.size(); i++)
	//	outputs[i].resize(10);
	//for (int l = 0; l < dataset.training_labels.size(); l++)
	//	outputs[l][dataset.training_labels[l]] = 1;
	//inputs.resize(dataset.training_images.size());
	//for (auto& i : inputs)
	//	i.resize(784);
	//for (int i = 0; i < inputs.size(); i++)
	//	for (int j = 0; j < 784; j++)
	//		inputs[i][j] = (double)dataset.training_images[i][j] / 255.0;
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
			if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Space) {
					net.propugate(inputs[0]);
				}
				if (event.key.code == sf::Keyboard::Escape) {
					window.close();
				}
				if (event.key.code == sf::Keyboard::LControl) {
					learn = !learn;
				}
				if (event.key.code == sf::Keyboard::LShift) {
					render = !render;
				}
			}
			if (button2.isClicked(window)) {
				render = true;
			}
			if (button.isClicked(window)) {
				learn = true;
			}
			if (event.type == sf::Event::MouseButtonPressed && sf::Keyboard::isKeyPressed(sf::Keyboard::X)) {
				if (event.mouseButton.button == sf::Mouse::Left) {
					inputs.push_back({ static_cast<double>((event.mouseButton.x - 600) / img_w - 1), static_cast<double>((event.mouseButton.y - 10) / img_h - 1) });
					outputs.push_back({ 1, 0 });
				}
				if (event.mouseButton.button == sf::Mouse::Right) {
					inputs.push_back({ static_cast<double>((event.mouseButton.x - 600) / img_w - 1), static_cast<double>((event.mouseButton.y - 10) / img_h - 1) });
					outputs.push_back({ 0, 1 });
				}
			}
			slider.handleEvent(event, window);
		}
		button.update(window);
		button2.update(window);
		net.propugate({ 1, 0 });
		if (learn) {
			if (!inputs.empty()) {
				net.train_batch(inputs, outputs, 0.8, 8);
				loss = net.cost(inputs, outputs);
			}

		}
		window.clear(sf::Color(100, 100, 155));
		button.draw(window);
		button2.draw(window);
#pragma region network_rendering

			float startX = 100.0f;
			sf::Text text;
			text.setFont(font);
			text.setString(std::string("error: ") + std::to_string(loss).c_str()); // limit digits
			text.setCharacterSize(32);
			text.setFillColor(sf::Color::White);
			text.setOrigin(text.getLocalBounds().width / 2, text.getLocalBounds().height / 2);
			text.setPosition(100, 100); // offset slightly above center
			window.draw(text);
			text.setString("right(blue)/left(red) click while holding x to place inputs"); // limit digits
			text.setPosition(1010, 200); 
			window.draw(text);

			for (size_t l = 0; l < net.layers.size() + 1; l++) {
				int n_nodes = (l == 0 ? layersizes[0] : net.layers[l - 1].n_nodes_out);

				float startY = (window.getSize().y - (n_nodes - 1) * ySpacing) / 2.0f;

				for (int i = 0; i < n_nodes; i++) {
					sf::CircleShape neuron(radius);
					neuron.setFillColor(activationColor(net.activations[l][i]));
					sf::Vector2f pos(startX + l * xSpacing, startY + i * ySpacing);

					neuron.setOrigin(radius, radius);
					neuron.setPosition(startX + l * xSpacing, startY + i * ySpacing);
					window.draw(neuron);

					sf::Text text;
					text.setFont(font);
					text.setString(std::to_string(net.activations[l][i]).substr(0, 4)); // limit digits
					text.setCharacterSize(14);
					text.setFillColor(sf::Color::Black);
					text.setOrigin(text.getLocalBounds().width / 2, text.getLocalBounds().height / 2);
					text.setPosition(pos.x, pos.y - 8); // offset slightly above center
					window.draw(text);

					 //connections to next layer
					if (l < net.layers.size()) {
						int n_next = net.layers[l].n_nodes_out;
						float nextStartY = (window.getSize().y - (n_next - 1) * ySpacing) / 2.0f;
						for (int j = 0; j < n_next; j++) {
							sf::Vertex line[] = {
								sf::Vertex(neuron.getPosition(), sf::Color(100,100,255)),
								sf::Vertex(sf::Vector2f(startX + (l + 1) * xSpacing, nextStartY + j * ySpacing), sf::Color(100,100,255))
							};
							window.draw(line, 2, sf::Lines);
						}
					}
				}
			}
#pragma endregion
#pragma region data_rendering
		if (render) {
			for (int i = 0; i < (int)img_h; i++) {
				for (int j = 0; j < (int)img_w; j++) {
					
					auto b = net.propugate({ static_cast<double>(j / img_w - 1.f), static_cast<double>(i / img_h -1.f) });
					
					if (b[0] > b[1]) {
						img.setPixel(j, i, sf::Color::Cyan);
					}
					else {
						img.setPixel(j, i, sf::Color::Red);

					}
				}
			}
			tex.update(img);

			render = false;
		}
		window.draw(sp);
		sf::CircleShape point(10);
		point.setOrigin(10, 10);

		for (int i = 0; i < inputs.size(); i++) {
			point.setPosition((inputs[i][0] + 1) * img_w + 600, (inputs[i][1] + 1) * img_h + 10);
			point.setFillColor(sf::Color::Red);
			if (outputs[i][0] > 0)
				point.setFillColor(sf::Color::Cyan);

			window.draw(point);
		}
#pragma endregion

		window.display();
	}

	return 0;
}