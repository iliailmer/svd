// TODO: add matrix multiplication for verification of correctness
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
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
  Matrix mat(rows, Vector(cols));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      file >> mat[i][j];
    }
  }

  return mat;
};

void eye(Matrix &M)
{
  for (size_t row_idx = 0; row_idx < M.size(); row_idx++) {
    for (size_t col_idx = 0; col_idx < M[0].size(); col_idx++) {
      if (row_idx == col_idx) {
        M[row_idx][col_idx] = 1.0;
      }
    }
  }
}
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

Matrix transpose(const Matrix &M)
{
  int m = M.size();
  int n = M[0].size();
  Matrix T(n, Vector(m, 0.0));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      T[j][i] = M[i][j];
    }
  }
  return T;
}

void bidiagonalize(Matrix &A, Matrix &U, Matrix &V)
{
  // Algorithm 1.a from
  // https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf
  int rows = A.size();
  int columns = A[0].size();
  int k = min(rows, columns);
  eye(U);
  eye(V);

  for (int i = 0; i < k; i++) {
    if (i < rows - 1) {
      Vector x(rows - i);
      for (int j = i; j < rows; j++) {
        x[j - i] = A[j][i];
      }
      Vector v = housholder_reflections(x);

      apply_householder_left(U, v, i, 0);
      apply_householder_left(A, v, i, i);
    }
    if (i < columns - 2) {

      Vector x(columns - i - 1);
      for (int j = i + 1; j < columns; j++) {
        x[j - i - 1] = A[i][j];
      }
      Vector v = housholder_reflections(x);

      apply_householder_right(V, v, 0, i + 1);
      apply_householder_right(A, v, i, i + 1);
    }
  }
  U = transpose(U);
}

void display_simple(Matrix A)
{
  for (const auto &row : A) {
    for (double val : row) {
      cout << val << " ";
    }
    cout << "\n";
  }
}
void display(const Matrix &A, double eps = 1e-10)
{
  int m = A.size();
  int n = A[0].size();

  // Find the width needed for each column
  vector<int> col_widths(n, 0);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      ostringstream oss;
      if (abs(A[i][j]) < eps) {
        oss << "0";
      }
      else {
        oss << fixed << setprecision(4) << A[i][j];
      }
      col_widths[j] = max(col_widths[j], (int)oss.str().length());
    }
  }

  // Print with proper alignment
  for (int i = 0; i < m; i++) {
    cout << "[ ";
    for (int j = 0; j < n; j++) {
      if (abs(A[i][j]) < eps) {
        cout << setw(col_widths[j]) << "0";
      }
      else {
        cout << setw(col_widths[j]) << fixed << setprecision(4) << A[i][j];
      }
      if (j < n - 1)
        cout << "  ";
    }
    cout << " ]\n";
  }
  cout << "\n";
}

int main(int argc, char *argv[])
{
  Matrix A = load_data("input.txt");
  int rows = A.size();
  int columns = A[0].size();
  Matrix U(rows, Vector(rows, 0.0));

  Matrix V(columns, Vector(columns, 0.0));

  bidiagonalize(A, U, V);
  display(A);

  cout << endl;
  display(U);
  cout << endl;
  display(V);
  return 0;
}
