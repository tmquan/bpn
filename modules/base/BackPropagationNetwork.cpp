#include "BackPropagationNetwork.hpp"

#include <stdio.h>
#include <stdlib.h>     
#include <string.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

using namespace std;
//----------------------------------------------------------------------------
#define at(x, y, z, dimx, dimy, dimz) (z*dimy*dimx+y*dimx+x)	
//----------------------------------------------------------------------------
BackPropagationNetwork::BackPropagationNetwork()
{}
//----------------------------------------------------------------------------
BackPropagationNetwork::BackPropagationNetwork(
	int 		layerSizes[], 
	const int 	lengthOfLayerSize, 
	int	 		layerFunctions[])
{
	//Retrieve the layer information
	m_layerCount = lengthOfLayerSize - 1; //Not count the input layer
	
	m_layerShape = new int[lengthOfLayerSize];
	memcpy(m_layerShape, layerSizes, lengthOfLayerSize*sizeof(int));
	
	//Input/Output from the previous run
	m_layerInput 	= NULL;
	m_layerOutput 	= NULL;
	
	// vector<float> v_weights;
	int layerMaximum = 0;
	for (int layer=0; layer<m_layerCount; layer++)
	{
		if(m_layerShape[layer] > layerMaximum)
			layerMaximum = m_layerShape[layer];
	}
	srand(time(NULL));
	
	m_weights = new float[m_layerCount*(layerMaximum+1)*(layerMaximum+1)];
	for (int layer=0; layer<m_layerCount; layer++)
	{
		int layerFirst 	= layer;
		int layerSecond = layer+1;
		int sizeFirst   = m_layerShape[layerFirst]+1; // For bias
		int sizeSecond  = m_layerShape[layerSecond];
		for(int i=0; i<sizeSecond; i++)
		{
			for(int j=0; j<sizeFirst; j++)
			{
				// This will generate a number from 0.0 to 1.0, inclusive.
				float randFloat = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				// v_weights.push_back(randFloat);
				m_weights[at(j, i, layer, sizeSecond, sizeFirst, m_layerCount)] = randFloat - 0.5;
			}
		}
	}
}
//----------------------------------------------------------------------------
BackPropagationNetwork::~BackPropagationNetwork()
{}
//----------------------------------------------------------------------------
void BackPropagationNetwork::printShape()
{
	printf("Shape of the network :\n");
	for(int layer=0; layer<m_layerCount+1; layer++)
		printf("\t%d", m_layerShape[layer]);
	printf("\n");
}
//----------------------------------------------------------------------------
void BackPropagationNetwork::printWeights()
{
	printf("Weights of the network :\n");
	for (int layer=0; layer<m_layerCount; layer++)
	{
		int layerFirst 	= layer;
		int layerSecond = layer+1;
		int sizeFirst   = m_layerShape[layerFirst]+1; // For bias
		int sizeSecond  = m_layerShape[layerSecond];
		for(int i=0; i<sizeSecond; i++)
		{
			for(int j=0; j<sizeFirst; j++)
			{
				printf("\t%4.6f", m_weights[at(j, i, layer, sizeSecond, sizeFirst, m_layerCount)] );	
			}
			printf("\n");
		}
		printf("\n");
	}
}
//----------------------------------------------------------------------------
