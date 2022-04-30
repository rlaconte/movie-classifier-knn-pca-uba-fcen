#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(SparseMatrix X);
	int newAlpha(unsigned int anuevo);
private:

	unsigned int alpha;
	Matrix autovectores;
};
