#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 5) {
		cerr << "Must pass four and only four values!" << endl;
		exit(-1);
	}

	double val1 = atof(argv[1]);
	double val2 = atof(argv[2]);
	double val3 = atof(argv[3]);
	double val4 = atof(argv[4]);

	cout << fabs(val1)+fabs(val2)+fabs(val3)+fabs(val4) << endl;

	return 0;

}
