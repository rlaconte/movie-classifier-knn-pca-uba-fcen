#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;

// Pasamos el alpha máximo que vamos a testear, y calculamos su matriz de autovectores
// Pasar a un alpha más chico de la misma base de entrenamiento no es mas que recortar columnas de
//    la matriz de autovectores
PCA::PCA(unsigned int n_components): alpha(n_components) {}

void PCA::fit(Matrix X){
	//fit recibe matrix X y devuelve la matriz de autovectores de la matriz de covarianza
	
	X.rowwise() -= X.colwise().mean();

	Matrix cov = X.transpose() * X;
	cov /= X.size()-1; 
	this->autovectores = get_first_eigenvalues(cov,this->alpha).second;
}

MatrixXd PCA::transform(SparseMatrix X){
	MatrixXd aux = MatrixXd(X);
	Matrix cambioDeBase = this->autovectores;
	aux = aux * cambioDeBase;
	return aux;
}

// Como precondicion, toma un alpha mas chico que ya tiene el pca en private. Deberiamos recortar la matriz de autovectores
// Se hace antes de hacer un nuevo transform
int PCA::newAlpha(unsigned int anuevo){ 
	// Nos quedamos solo con los primeros anuevo autovectores
	if(anuevo > this->alpha){
		cerr << "anuevo debe ser menor que aviejo" << endl;
		return -1;
	}
	Matrix avecNuevos = (this->autovectores)(Eigen::all, Eigen::seq(0,anuevo-1));
	this->autovectores = avecNuevos;
	this->alpha = anuevo;
	// Autovectores debería tener tamaño (***,anuevo) despues de esto, y el nuevo alpha debería ser anuevo
	cout << "nuevo alpha: " << this->alpha << " nueva cant avec: " << this->autovectores.cols() << endl;
	return 0;
}