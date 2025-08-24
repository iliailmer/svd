#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;
Matrix load_data(const string &path_to_input)
{
  string line;
  ifstream file(path_to_input);
  size_t rows, cols;
  file >> rows >> cols;
  Matrix mat(rows, vector<double>(cols));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      file >> mat[i][j];
    }
  }

  return mat;
};

int sign(float x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }
double norm(Vector x)
{
  double norm_x = 0.0;
  for (auto each : x) {
    norm_x = each * each + norm_x;
  }
  norm_x = sqrt(norm_x);
  return norm_x;
}

Vector housholder_reflections(const Vector &x)
{
  int n = x.size();
  Vector v(n, 0.0);  // output
  Vector e1(n, 0.0); // basis vector
  e1[0] = 1.0;
  double norm_x = norm(x);
  v = x;
  v[0] = v[0] + sign(x[0]) * norm_x;
  double norm_v = norm(v);
  for (int i = 0; i < v.size(); i++) {
    v[i] = v[i] / norm_v;
  }

  return v;
}

int main(int argc, char *argv[])
{
  Matrix A = load_data("input.txt");
  for (const auto &row : A) {
    for (double val : row) {
      cout << val << " ";
    }
    cout << "\n";
  }
  return 0;
}
