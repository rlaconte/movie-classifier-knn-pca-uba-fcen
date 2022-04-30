#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    unsigned i = 0;
    Vector aux;
    while (i < num_iter)
    {
        aux = b;
    	Vector prod = X * b; 
    	prod /= prod.norm();
    	b = prod;
        // ACÁ se cambia .norm por .lpNorm<Eigen::Infinity>() o por .lpNorm<1>() para testear otras normas
        if(((b-aux).norm()) < eps){
            break;
        }
    	i++;
    }
    // Para chequear que el codigo está funcionando y ver cuándo hace el corte
    cout << "Hice " << i << " iteraciones" << endl;

    eigenvalue = b.transpose() * X * b; 
    eigenvalue /= b.transpose() * b;

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    unsigned i = 0;
    while (i < num)
    {
    	pair<double, Vector> powerIt = power_iteration(A, num_iter, epsilon);
    	eigvalues(i) = powerIt.first;
    	eigvectors(Eigen::all,i) = (powerIt.second).transpose();
    	A -= powerIt.first * (powerIt.second * powerIt.second.transpose());
    	i++;
    }

    return make_pair(eigvalues, eigvectors);
}

