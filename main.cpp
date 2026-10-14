#include <algorithm>
#include <cstddef>
#include <cstdlib>
#define EPS 1e-15
#define MAX_ITER 500
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;
void display(const Matrix &A, double eps = 1e-10) {
  size_t m = A.size();
  size_t n = A[0].size();

  // Find the width needed for each column
  vector<int> col_widths(n, 0);
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      ostringstream oss;
      if (abs(A[i][j]) < eps) {
        oss << "0";
      } else {
        oss << fixed << setprecision(4) << A[i][j];
      }
      col_widths[j] = max(col_widths[j], (int)oss.str().length());
    }
  }

  // Print with proper alignment
  for (size_t i = 0; i < m; i++) {
    cout << "[ ";
    for (size_t j = 0; j < n; j++) {
      if (abs(A[i][j]) < eps) {
        cout << setw(col_widths[j]) << "0";
      } else {
        cout << setw(col_widths[j]) << fixed << setprecision(4) << A[i][j];
      }
      if (j < n - 1)
        cout << "  ";
    }
    cout << " ]\n";
  }
  cout << "\n";
}

Matrix load_data(const string &path_to_input) {
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

void eye(Matrix &M) {
  for (size_t row_idx = 0; row_idx < M.size(); row_idx++) {
    for (size_t col_idx = 0; col_idx < M[0].size(); col_idx++) {
      if (row_idx == col_idx) {
        M[row_idx][col_idx] = 1.0;
      }
    }
  }
}
int sign(double x) { return (x >= 0) ? 1 : ((x < 0) ? -1 : 0); }

struct GivensRotation {
  double _cos, _sin;
  size_t row_i, row_j;

  GivensRotation(double a, double b, size_t row_i, size_t row_j)
      : row_i(row_i), row_j(row_j) {
    if (abs(b) < EPS) {
      _cos = 1.0;
      _sin = 0.0;
    } else if (abs(a) < EPS) {
      _cos = 0.0;
      _sin = 1.0;
    } else {
      if (abs(a) > abs(b)) {
        double t = b / a;
        double u = sign(a) * sqrt(1 + t * t);
        _cos = 1 / u;
        _sin = _cos * t;
      } else {
        double t = a / b;
        double u = sign(b) * sqrt(1 + t * t);
        _sin = 1 / u;
        _cos = t / u;
      }
    }
  }

  void apply_left(Matrix &M) {
    for (size_t j = 0; j < M[0].size(); j++) {
      double tmp = _cos * M[row_i][j] + _sin * M[row_j][j];
      M[row_j][j] = -_sin * M[row_i][j] + _cos * M[row_j][j];
      M[row_i][j] = tmp;
    }
  }
  void apply_right(Matrix &M) {
    for (size_t j = 0; j < M.size(); j++) {
      double tmp = _cos * M[j][row_i] + _sin * M[j][row_j];
      M[j][row_j] = -_sin * M[j][row_i] + _cos * M[j][row_j];
      M[j][row_i] = tmp;
    }
  }
};

double norm(const Vector &x) {
  double norm_x = 0.0;
  for (auto each : x) {
    norm_x = each * each + norm_x;
  }
  norm_x = sqrt(norm_x);
  return norm_x;
}

double mnorm(const Matrix &x) { // naive matrix norm
  double _norm = 0.0;
  for (auto each : x) {
    auto norm_vec = norm(each);
    _norm += norm_vec * norm_vec;
  }
  _norm = sqrt(_norm);
  return _norm;
}

auto vdiff(Vector a, Vector b) {
  size_t size_a = a.size();
  size_t size_b = b.size();
  Vector _diff = Vector(size_a, 0.0);
  if (size_a != size_b) {
    cout << "ERROR in vdiff: Matrix sizes do not match" << endl;
    exit(1);
  }
  for (auto [a_i, b_i] : std::views::zip(a, b)) {
    _diff.push_back(abs(a_i - b_i));
  }
  return _diff;
}

auto mdiff(Matrix a, Matrix b) { // naive matrix difference
  size_t n_rows = a.size();
  size_t n_cols = a[0].size();
  Matrix _diff = Matrix(n_rows, Vector(n_cols, 0.0));
  for (auto [a_i, b_i] : std::views::zip(a, b)) {
    _diff.push_back(vdiff(a_i, b_i));
  }
  return _diff;
}

Matrix mmult(Matrix A, Matrix B) {
  // NOTE: slow implementation of matrix multiplication
  size_t rows_A = A.size();
  size_t cols_A = A[0].size();

  size_t rows_B = B.size();
  size_t cols_B = B[0].size();

  if (cols_A != rows_B) {
    cout << "ERROR: Matrix sizes do not match" << endl;
    cout << "A: " << rows_A << "x" << cols_A << endl;
    cout << "B: " << rows_B << "x" << cols_B << endl;
    exit(1);
  }
  Matrix C(rows_A, Vector(cols_B, 0.0));

  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_A; j++) {
      for (size_t k = 0; k < cols_B; k++) {
        C[i][k] = C[i][k] + A[i][j] * B[j][k];
      }
    }
  }
  return C;
}

Vector housholder_reflections(const Vector &x) {
  size_t n = x.size();
  Vector v(n, 0.0);  // output
  Vector e1(n, 0.0); // basis vector
  e1[0] = 1.0;
  double norm_x = norm(x);
  v = x;
  v[0] = v[0] + sign(x[0]) * norm_x;
  double norm_v = norm(v);
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = v[i] / norm_v;
  }

  return v;
}

Vector matrix_vector_multiply_left(Matrix A, const Vector &v, size_t start_row,
                                   size_t start_col) {
  size_t n = A[0].size();
  Vector result(n - start_col, 0.0);
  for (size_t row_idx = start_row; row_idx < A.size(); row_idx++) {
    for (size_t col_idx = start_col; col_idx < A[0].size(); col_idx++) {
      result[col_idx - start_col] =
          result[col_idx - start_col] +
          A[row_idx][col_idx] * v[row_idx - start_row];
    }
  }
  return result;
}

Vector matrix_vector_multiply_right(Matrix A, const Vector &v, size_t start_row,
                                    size_t start_col) {
  size_t n = A.size();
  Vector result(n, 0.0);
  for (size_t row_idx = start_row; row_idx < A.size(); row_idx++) {
    for (size_t col_idx = start_col; col_idx < A[0].size(); col_idx++) {
      result[row_idx - start_row] =
          result[row_idx - start_row] +
          A[row_idx][col_idx] * v[col_idx - start_col];
    }
  }
  return result;
}

void apply_householder_left(Matrix &A, const Vector &v, size_t start_row,
                            size_t start_col) {
  size_t m = A.size();
  size_t n = A[0].size();

  Vector w = matrix_vector_multiply_left(A, v, start_row, start_col);

  for (size_t row_idx = start_row; row_idx < m; row_idx++) {
    for (size_t col_idx = start_col; col_idx < n; col_idx++) {
      A[row_idx][col_idx] = A[row_idx][col_idx] -
                            2 * v[row_idx - start_row] * w[col_idx - start_col];
    }
  }
}

void apply_householder_right(Matrix &A, const Vector &v, size_t start_row,
                             size_t start_col) {
  size_t m = A.size();
  size_t n = A[0].size();

  Vector w = matrix_vector_multiply_right(A, v, start_row, start_col);

  for (size_t row_idx = start_row; row_idx < m; row_idx++) {
    for (size_t col_idx = start_col; col_idx < n; col_idx++) {
      A[row_idx][col_idx] = A[row_idx][col_idx] -
                            2 * v[col_idx - start_col] * w[row_idx - start_row];
    }
  }
}

Matrix transpose(const Matrix &M) {
  size_t m = M.size();
  size_t n = M[0].size();
  Matrix T(n, Vector(m, 0.0));
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      T[j][i] = M[i][j];
    }
  }
  return T;
}

void bidiagonalize(Matrix &A, Matrix &U, Matrix &V) {
  // Algorithm 1.a from
  // https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf
  size_t rows = A.size();
  size_t columns = A[0].size();
  size_t k = min(rows, columns);
  eye(U);
  eye(V);

  for (size_t i = 0; i < k; i++) {
    if (i < rows - 1) {
      Vector x(rows - i);
      for (size_t j = i; j < rows; j++) {
        x[j - i] = A[j][i];
      }
      Vector v = housholder_reflections(x);

      apply_householder_left(U, v, i, 0);
      apply_householder_left(A, v, i, i);
    }
    if (i < columns - 2) {

      Vector x(columns - i - 1);
      for (size_t j = i + 1; j < columns; j++) {
        x[j - i - 1] = A[i][j];
      }
      Vector v = housholder_reflections(x);

      apply_householder_right(V, v, 0, i + 1);
      apply_householder_right(A, v, i, i + 1);
    }
  }
}

void display_simple(const Matrix &A) {
  for (const auto &row : A) {
    for (double val : row) {
      cout << val << " ";
    }
    cout << "\n";
  }
}
struct SVD {
  Matrix U;
  Matrix S;
  Matrix V;

  SVD(Matrix U, Matrix S, Matrix V) : U(U), S(S), V(V) {};
};

SVD Golub_Reisch_SVD(Matrix A, double eps) {

  int iter_count = 0;
  Matrix B = A;
  size_t rows = A.size();
  size_t columns = A[0].size();
  Matrix U(rows, Vector(rows, 0.0));
  Matrix S(rows, Vector(columns, 0.0));
  Matrix V(columns, Vector(columns, 0.0));

  bidiagonalize(B, U, V);
  display(B);
  while (true) {
    iter_count += 1;
    if (iter_count > MAX_ITER) {
      break;
    }

    cout << "\n=== Iteration " << iter_count << " ===" << endl;
    cout << "B at start:" << endl;
    display(B);

    for (size_t i = 0; i < columns - 1; i++) {
      if (abs(B[i][i + 1]) <= eps * (abs(B[i][i]) + abs(B[i + 1][i + 1]))) {
        B[i][i + 1] = 0.0;
      }
    }

    // performing block splitting
    size_t p = 0;
    while (p < columns - 1 && abs(B[p][p + 1]) < eps) {
      p++; // Count converged superdiagonals from left
    }

    size_t q = 0;
    for (size_t i = columns - 2; i >= p; i--) {
      if (abs(B[i][i + 1]) < eps) {
        q++; // Count converged superdiagonals from right
      } else {
        break;
      }
    }

    cout << "p = " << p << ", q = " << q << endl;
    cout << "Block size = " << (columns - p - q) << endl;

    if (p + q >= columns - 1) {
      cout << "CONVERGED!" << endl;
      for (size_t i = 0; i < columns; i++) {
        S[i][i] = abs(B[i][i]);
      }
      return SVD(U, S, V);
    }
    for (size_t i = p; i < columns - q - 1; i++) {
      if (abs(B[i][i]) < eps) {
        // apply givens rotations
        for (size_t k = i; k < min(columns - 1, rows - 1); k++) {
          // Zero out B[k][k+1] using B[k+1][k+1]
          if (k + 1 < columns) {
            GivensRotation rot(B[k + 1][k + 1], B[k][k + 1], k, k + 1);
            rot.apply_right(B);
            rot.apply_right(V);
          }
        }
        break;
      }
    }

    // wilkinson shift

    size_t block_size = columns - p - q;

    if (block_size <= 1) {
      // Block is already converged
      for (size_t i = 0; i < min(rows, columns); i++) {
        S[i][i] = B[i][i];
      }
      return SVD(U, S, V);
    }
    double alpha = B[columns - q - 2][columns - q - 2];
    double beta = B[columns - q - 2][columns - q - 1];
    double gamma = B[columns - q - 1][columns - q - 1];

    Matrix C(2, Vector(2, 0.0));
    C[0][0] = alpha * alpha + beta * beta;
    C[0][1] = beta * gamma;
    C[1][0] = beta * gamma;
    C[1][1] = gamma * gamma;

    double a = C[0][0];
    double b = C[0][1];
    double c = C[1][1];

    double d = (a - c) / 2.0;
    double mu = c - sign(d) * b * b / (abs(d) + sqrt(d * d + b * b));
    size_t k = p;
    alpha = B[k][k] * B[k][k] - mu;
    beta = B[k][k] * B[k][k + 1];
    for (size_t _k = p; _k < columns - q - 1; _k++) {
      // givens rotations
      GivensRotation rot_1 = GivensRotation(alpha, beta, _k, _k + 1);
      rot_1.apply_right(B);
      rot_1.apply_right(V);

      alpha = B[_k][_k];
      beta = B[_k + 1][_k];

      GivensRotation rot_2 = GivensRotation(alpha, beta, _k, _k + 1);
      rot_2.apply_left(B);
      rot_2.apply_left(U);

      if (k < columns - q - 2) {

        alpha = B[_k][_k + 1];
        beta = B[_k][_k + 2];
      }
    }

    cout << "B at end of iteration:" << endl;
    display(B);
  }

  return SVD(U, S, V);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <input_file>" << endl;
    return 1;
  }

  Matrix A = load_data(argv[1]);
  SVD svd_res = Golub_Reisch_SVD(A, EPS);

  // restore original
  Matrix A_restored;
  A_restored = mmult(svd_res.S, transpose(svd_res.V));
  A_restored = mmult(transpose(svd_res.U), A_restored);
  double A_err = mnorm(mdiff(A, A_restored));
  display(A_restored);
  cout << A_err << std::endl;

  return 0;
}
