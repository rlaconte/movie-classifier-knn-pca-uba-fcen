#pragma once

#include "types.h"


class KNNClassifier {
public:
	KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);

    unsigned int vecinos();
	MatrixXd dame_X();
	Matrix dame_y();
	Vector predictNewK(unsigned int knuevo);
private:
	unsigned int _n_neighbors;
	SparseMatrix _X;
	Matrix _y;
	// Matriz con los votos de los primeros k vecinos
	// es sparse ya que se guarda el voto como 1 (positivo) o 0 (negativo)
	// y la distribución debería ser aproximadamente 50/50
	Eigen::SparseMatrix<double,Eigen::ColMajor> _vote_mat;
	
	void predict_row(Vector row, unsigned k);
	Vector distance_to_row(Vector row);
};
