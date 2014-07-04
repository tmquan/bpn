#ifndef __BACKPROPAGATIONNETWORK_HPP
#define __BACKPROPAGATIONNETWORK_HPP

#include <iostream>
#include <vector>
#include <list>

using namespace std;
//----------------------------------------------------------------------------
class BackPropagationNetwork
{
private:
	int 					m_layerCount;
	int* 					m_layerShape;
	int*					m_transferFuncs; 		//List of transfer functions
	float*					m_weights;
	float*					m_layerInput; 		//Data from previous run
	float*					m_layerOutput; 		//Data from previous run
	
public:
	// Default Constructor
	BackPropagationNetwork();
	
	// Overload Constructor
	BackPropagationNetwork(int 			layerSizes[], 
						   const int 	lengthOfLayerSize, 
						   int	 		layerFunctions[]);
	
	// Destructor
	~BackPropagationNetwork();
	
	void printShape();
	void printWeights();
};
//----------------------------------------------------------------------------


#endif