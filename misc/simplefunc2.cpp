#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
	while (1) {
		string extra;
		double val1;
		double val2;
		double val3;
		double val4;
		bool success = true;
		if ((cin >> val1)) {
		}
		else
			success = false;
		if ((cin >> val2)) {
		}
		else
			success = false;
		if ((cin >> val3)) {
		}
		else
			success = false;
		if ((cin >> val4)) {
		}
		else
			success = false;
		if (success) {
			cout << fabs(val1)+fabs(val2)+fabs(val3)+fabs(val4) << endl;
		}
		else {
			cout << "Quitting!" << endl;
			exit(-1);
		}
	}

	return 0;

}
