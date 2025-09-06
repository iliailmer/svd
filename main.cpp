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

Vector matrix_vector_multiply_left(Matrix A, const Vector v, int start_row,
                                   int start_col)
{
  int n = A[0].size();
  Vector result(n - start_col, 0.0);
  for (int row_idx = start_row; row_idx < A.size(); row_idx++) {
    for (int col_idx = start_col; col_idx < A[0].size(); col_idx++) {
      result[col_idx - start_col] =
          result[col_idx - start_col] +
          A[row_idx][col_idx] * v[row_idx - start_row];
    }
  }
  return result;
}

Vector matrix_vector_multiply_right(Matrix A, const Vector v, int start_row,
                                    int start_col)
{
  int n = A.size();
  Vector result(n, 0.0);
  for (int row_idx = start_row; row_idx < A.size(); row_idx++) {
    for (int col_idx = start_col; col_idx < A[0].size(); col_idx++) {
      result[row_idx - start_row] =
          result[row_idx - start_row] +
          A[row_idx][col_idx] * v[col_idx - start_col];
    }
  }
  return result;
}

void apply_householder_left(Matrix &A, const Vector &v, int start_row,
                            int start_col)
{
  int m = A.size();
  int n = A[0].size();

  Vector w = matrix_vector_multiply_left(A, v, start_row, start_col);

  for (int row_idx = start_row; row_idx < m; row_idx++) {
    for (int col_idx = start_col; col_idx < n; col_idx++) {
      A[row_idx][col_idx] = A[row_idx][col_idx] -
                            2 * v[row_idx - start_row] * w[col_idx - start_col];
    }
  }
}

void apply_householder_right(Matrix &A, const Vector &v, int start_row,
                             int start_col)
{
  int m = A.size();
  int n = A[0].size();

  Vector w = matrix_vector_multiply_right(A, v, start_row, start_col);

  for (int row_idx = start_row; row_idx < m; row_idx++) {
    for (int col_idx = start_col; col_idx < n; col_idx++) {
      A[row_idx][col_idx] = A[row_idx][col_idx] -
                            2 * v[col_idx - start_col] * w[row_idx - start_row];
    }
  }
}
void bidiagonalize(Matrix &A)
{
  // Algorithm 1.a from
  // https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf
  int rows = A.size();
  int columns = A[0].size();
  int k = min(rows, columns);

  for (int i = 0; i < k; i++) {
    if (i < rows - 1) {
      Vector x(rows - i);
      for (int j = i; j < rows; j++) {
        x[j - i] = A[j][i];
      }
      Vector v = housholder_reflections(x);

      apply_householder_left(A, v, i, i);
    }
    if (i < columns - 2) {

      Vector x(columns - i - 1);
      for (int j = i + 1; j < columns; j++) {
        x[j - i - 1] = A[i][j];
      }
      Vector v = housholder_reflections(x);

      apply_householder_right(A, v, i, i + 1);
    }
  }
}

void display(Matrix A)
{
  for (const auto &row : A) {
    for (double val : row) {
      cout << val << " ";
    }
    cout << "\n";
  }
}

int main(int argc, char *argv[])
{
  Matrix A = load_data("input.txt");
  bidiagonalize(A);
  display(A);
  return 0;
}
