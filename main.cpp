#define EPS 1e-15
#include <cmath>
#include <fstream>
#include <iomanip>
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

struct GivensRotation {
  double _cos, _sin;
  int row_i, row_j;

  GivensRotation(double a, double b, int row_i, int row_j)
      : row_i(row_i), row_j(row_j)
  {
    if (abs(a) < EPS) {
      _cos = 1.0;
      _sin = 0.0;
    }
    else if (abs(b) < EPS) {
      _cos = 0.0;
      _sin = 1.0;
    }
    else {
      double r = sqrt(a * a + b * b);
      _cos = a / r;
      _sin = b / r;
    }
  }

  void apply_left(Matrix &M)
  {
    for (int j = 0; j < M[0].size(); j++) {
      double tmp = _cos * M[row_i][j] - _sin * M[row_j][j];
      M[row_j][j] = _sin * M[row_i][j] + _cos * M[row_j][j];
      M[row_i][j] = tmp;
    }
  }
  void apply_right(Matrix &M)
  {
    for (int j = 0; j < M.size(); j++) {
      double tmp = _cos * M[j][row_i] - _sin * M[j][row_j];
      M[row_j][j] = _sin * M[j][row_i] + _cos * M[j][row_j];
      M[row_i][j] = tmp;
    }
  }
};

double norm(Vector x)
{
  double norm_x = 0.0;
  for (auto each : x) {
    norm_x = each * each + norm_x;
  }
  norm_x = sqrt(norm_x);
  return norm_x;
}

Matrix mmult(Matrix A, Matrix B)
{
  // NOTE: slow implementation of matrix multiplication
  int rows_A = A.size();
  int cols_A = A[0].size();

  int rows_B = B.size();
  int cols_B = B[0].size();

  if (cols_A != rows_B) {
    cout << "ERROR: Matrix sizes do not match" << endl;
    cout << "A: " << rows_A << "x" << cols_A << endl;
    cout << "B: " << rows_B << "x" << cols_B << endl;
    exit(1);
  }
  Matrix C(rows_A, Vector(cols_B, 0.0));

  for (int i = 0; i < rows_A; i++) {
    for (int j = 0; j < cols_A; j++) {
      for (int k = 0; k < cols_B; k++) {
        C[i][k] = C[i][k] + A[i][j] * B[j][k];
      }
    }
  }
  return C;
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

struct SVD {
  Matrix U;
  Matrix S;
  Matrix V;

  SVD(Matrix U, Matrix S, Matrix V) : U(U), S(S), V(V) {};
};

SVD Golub_Reisch_SVD(Matrix A, double eps)
{

  Matrix B = A;
  int rows = A.size();
  int columns = A[0].size();
  Matrix U(rows, Vector(rows, 0.0));
  Matrix S(columns, Vector(columns, 0.0));
  Matrix V(columns, Vector(columns, 0.0));

  bidiagonalize(B, U, V);
  display(B);
  while (true) {

    for (int i = 0; i < columns - 1; i++) {
      if (abs(B[i][i + 1]) < eps * abs(B[i][i] + B[i + 1][i + 1])) {
        B[i][i + 1] = 0.0;
      }
    }
    // performing block splitting
    int p = 0;
    while (p < columns - 1 && abs(B[p][p + 1]) < eps) {
      p++; // Count converged superdiagonals from left
    }

    int q = 0;
    for (int i = columns - 2; i >= p; i--) {
      if (abs(B[i][i + 1]) < eps) {
        q++; // Count converged superdiagonals from right
      }
      else {
        break;
      }
    }
    cout << "(p, q)" << endl;
    cout << "(" << p << "," << q << ")" << endl;
    if (p + q >= columns - 1) {
      for (int i = 0; i < rows; i++) {
        S[i][i] = B[i][i];
      }
      return SVD(U, S, V);
    }

    // TODO: Add Golub-Kahan QR step here
    // For now, break to avoid infinite loop
    break;
  }

  return SVD(U, S, V);
}

int main(int argc, char *argv[])
{
  Matrix A = load_data("input.txt");
  int rows = A.size();
  int columns = A[0].size();
  SVD svd_res = Golub_Reisch_SVD(A, 1e-4);
  display(svd_res.S);

  return 0;
}
