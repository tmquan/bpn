#include <iostream>
#include "dnn.hpp"

using namespace std;

int main(int argc, char *argv[])
{
	int layerSize[3] = {3,4,2};
	BackPropagationNetwork bpn(layerSize, 3, NULL);
	
	bpn.printShape();
	bpn.printWeights();
}
