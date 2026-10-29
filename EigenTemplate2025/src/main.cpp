#include <iostream>
#include <random>
#include <Eigen/Sparse>

int main(int argc, char** argv)
{
	int n = 10;
	double tau = 1;

	// Naplnime koeficienty matice
	std::vector<Eigen::Triplet<double>> coefficients;
	coefficients.reserve(3 * (n - 2) + 2); // Rezervujeme miesto nenulove koeficienty matice
	for (size_t i = 0; i < n; i++)
	{
		if (i == 0 || i == n - 1) {
			coefficients.push_back(Eigen::Triplet<double>(i, i, 1));
		}
		else {
			coefficients.push_back(Eigen::Triplet<double>(i, i, 1 + 2 * tau));
			coefficients.push_back(Eigen::Triplet<double>(i, i - 1, -tau));
			coefficients.push_back(Eigen::Triplet<double>(i, i + 1, -tau));
		}
	}

	// Vytvorime maticu
	Eigen::SparseMatrix<double> A(n, n);
	A.setFromTriplets(coefficients.begin(), coefficients.end());

	// Vypis matice A
	std::cout << "Matrix A (dense format):\n";
	Eigen::MatrixXd denseA = Eigen::MatrixXd(A); // Konverzia na hustu maticu
	std::cout << denseA << "\n\n";

	std::cout << "Matrix A (sparse format):\n";
	for (int k = 0; k < A.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
		{
			std::cout << "(" << it.row() << ", " << it.col() << "): " << it.value() << "\n";
		}
	}

	// Definicia vektora pravej strany a vektora neznamych
	Eigen::VectorXd b(n);
	Eigen::VectorXd sol(n);

	// Naplnime pravu stranu
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(-0.5, 0.5);

	for (size_t i = 0; i < n; i++)
	{
		b[i] = i + 1;
	}

	for (size_t i = 1; i < n - 1; i++)
	{
		b[i] += distr(gen);
	}

	std::cout << "Prava strana" << std::endl;
	for (size_t i = 0; i < n; i++)
	{
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;

	//Riesenie pomocou interacneho solvera BiCGSTAB
	/*Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	solver.setMaxIterations(10000);
	solver.setTolerance(1E-8);
	solver.compute(A);
	sol = solver.solve(b);

	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error() << std::endl;*/

	//Riesenie pomocou priameho solvera SparseLU
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Decomposition failed!" << std::endl;
		return 0;
	}

	sol = solver.solve(b);

	// Vypis riesenia
	std::cout << "Riesenie" << std::endl;
	for (size_t i = 0; i < n; i++)
	{
		std::cout << sol[i] << " ";
	}

	//std::cout << "Solution:\n" << sol.transpose() << std::endl;

	return 0;
}