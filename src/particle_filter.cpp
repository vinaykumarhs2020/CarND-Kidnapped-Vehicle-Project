/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

// #define _DEBUG

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// Set number of particles:
	num_particles = 10;
	// Initialize x,y, psi distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	default_random_engine gen;

#ifdef _DEBUG
	std::cout << "X: " << x << " Y: " << y << " theta: " << theta << std::endl;
#endif

	// Initialize particles
	for(int i=0; i<num_particles; ++i){
		particles.push_back(Particle());
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

	is_initialized = true;
	#ifdef _DEBUG
		std::cout << "Filter initialized" << std::endl;
	#endif

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
#ifdef _DEBUG
	std::cout << "Prediction begin" << std::endl;
#endif

	// Create distributions
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	default_random_engine gen;

	// Update particles
	for(int i=0; i<num_particles; ++i){
		if(abs(yaw_rate) > 0.0001){ // Handle division by zero
			particles[i].x += velocity*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta))/yaw_rate + dist_x(gen);
			particles[i].y += velocity*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t))/yaw_rate + dist_y(gen);
			particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
		} else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
#ifdef _DEBUG
	std::cout << "Prediction End" << std::endl;
#endif
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

#ifdef _DEBUG
	std::cout << "Update Weights begin" << std::endl;
#endif

	// Clear the existing weights;
	weights.clear();

	// Loop through all the particles:
	for(int i=0; i<num_particles; ++i){
		// Clear previous sense and associations
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		particles[i].associations.clear();
		// Loop through all the observations
		for(int j=0; j< observations.size(); ++j){
			particles[i].sense_x.push_back(
				observations[j].x * cos(particles[i].theta)
				- observations[j].y * sin(particles[i].theta)
				+ particles[i].x);
			particles[i].sense_y.push_back(
				observations[j].x * sin(particles[i].theta)
				+ observations[j].y * cos(particles[i].theta)
				+ particles[i].y);
			// Create a temp container for distances
			vector<double> _dists(map_landmarks.landmark_list.size());
			// measure distances to all landmarks
			std::transform(
				map_landmarks.landmark_list.begin(),
				map_landmarks.landmark_list.end(),
				_dists.begin(),
				[&](Map::single_landmark_s _lm){
					return dist(
						_lm.x_f,
						_lm.y_f,
						particles[i].sense_x[j],
						particles[i].sense_y[j]);
				}
			);
#ifdef _DEBUG
	// std::cout << "_dists:" << std::endl;
	// for (auto d = _dists.begin(); d != _dists.end() ; d++) {
	// 	std::cout << *d << " ";
	// }
	// std::cout << std::endl;
#endif
			// Capture minimum element index in the _dists vector
			particles[i].associations.push_back(
				std::distance(
					std::begin(_dists),
					std::min_element(std::begin(_dists), std::end(_dists))
				) + 1
			);
#ifdef _DEBUG
		std::cout << "associations: " << particles[i].associations[j] << std::endl;
#endif
		}
	}
	// Calculate weights:
	for (size_t i = 0; i < num_particles; i++) {
		particles[i].weight = 1.0;
		for (size_t j = 0; j < observations.size(); j++) {
			auto _a = particles[i].associations[j];
			particles[i].weight *= 1.0/(2* M_PI * std_landmark[0] * std_landmark[1]) *
				exp(-(
					pow((particles[i].sense_x[j] - map_landmarks.landmark_list[_a - 1].x_f), 2) / (2.0 * pow(std_landmark[0], 2)) +
					pow((particles[i].sense_y[j] - map_landmarks.landmark_list[_a - 1].y_f), 2) / (2.0 * pow(std_landmark[1], 2))
				));
		}
		weights.push_back(particles[i].weight);
	}
#ifdef _DEBUG
	std::cout << "Update weights end" << std::endl;
#endif
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
#ifdef _DEBUG
	std::cout << "Resample begin" << std::endl;
#endif
	std::vector<Particle> _particles;
	// Use discrete_distribution (http://www.cplusplus.com/reference/random/discrete_distribution/)
	// to represent a distribution proportional to weights
	discrete_distribution<int> resample_dist(weights.begin(), weights.end());
	default_random_engine gen;
	for (size_t i = 0; i < num_particles; i++) {
		int ptcl = resample_dist(gen);
		_particles.push_back(particles[ptcl]);
	}
	particles = _particles;
#ifdef _DEBUG
	std::cout << "Resample end" << std::endl;
#endif
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
