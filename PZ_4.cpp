#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>

using namespace std;
using hr_clock = chrono::high_resolution_clock;
using seconds = chrono::duration<double>;

inline size_t idx(size_t i, size_t j, size_t N) { return i * N + j; }

void build_A(vector<double>& A, size_t N)
{
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[idx(i, j, N)] = (i == j) ? 100.0
            : 1.0 - 0.1 * static_cast<double>(i + 1)
            - 0.2 * static_cast<double>(j + 1);
}

void build_f(const vector<double>& A, vector<double>& f, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) s += A[idx(i, j, N)];
        f[i] = s; // потому что x* = (1,…,1)
    }
}

// LU
bool lu_decompose(vector<double>& A, vector<size_t>& piv, size_t N)
{
    piv.resize(N);
    for (size_t k = 0; k < N; ++k) {
        size_t max_row = k; double max_val = fabs(A[idx(k, k, N)]);
        for (size_t i = k + 1; i < N; ++i) {
            double v = fabs(A[idx(i, k, N)]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val == 0.0) return false; // вырождена
        piv[k] = max_row;
        if (max_row != k)
            for (size_t j = 0; j < N; ++j)
                swap(A[idx(k, j, N)], A[idx(max_row, j, N)]);
        for (size_t i = k + 1; i < N; ++i) {
            A[idx(i, k, N)] /= A[idx(k, k, N)];
            double lik = A[idx(i, k, N)];
            for (size_t j = k + 1; j < N; ++j)
                A[idx(i, j, N)] -= lik * A[idx(k, j, N)];
        }
    }
    return true;
}

void lu_solve(const vector<double>& LU, const vector<size_t>& piv,
    vector<double>& x, const vector<double>& b, size_t N)
{
    x = b;
    for (size_t k = 0; k < N; ++k)
        if (piv[k] != k) swap(x[k], x[piv[k]]);
    // Ly = Pb
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < i; ++j) x[i] -= LU[idx(i, j, N)] * x[j];
    // Ux = y
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        for (size_t j = i + 1; j < N; ++j) x[i] -= LU[idx(i, j, N)] * x[j];
        x[i] /= LU[idx(i, i, N)];
    }
}

// QR
inline void apply_givens(double& a, double& b, double c, double s)
{
    double t = c * a - s * b;
    b = s * a + c * b;
    a = t;
}

bool qr_givens_solve(const vector<double>& A_in, const vector<double>& f_in,
    vector<double>& x, size_t N)
{
    vector<double> R = A_in;
    vector<double> b = f_in;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = j + 1; i < N; ++i) {
            double a = R[idx(j, j, N)], bij = R[idx(i, j, N)];
            if (fabs(bij) < 1e-14) continue;
            double r = hypot(a, bij);
            double c = a / r, s = -bij / r;
            for (size_t k = j; k < N; ++k) {
                double Rjk = R[idx(j, k, N)], Rik = R[idx(i, k, N)];
                apply_givens(Rjk, Rik, c, s);
                R[idx(j, k, N)] = Rjk;
                R[idx(i, k, N)] = Rik;
            }
            apply_givens(b[j], b[i], c, s);
        }
    }
    for (size_t i = 0; i < N; ++i)
        if (fabs(R[idx(i, i, N)]) < 1e-15) return false;
    x.assign(N, 0.0);
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        double s = b[i];
        for (size_t j = i + 1; j < N; ++j) s -= R[idx(i, j, N)] * x[j];
        x[i] = s / R[idx(i, i, N)];
    }
    return true;
}

struct Stat { double avg_time{ 0.0 }; double delta{ 0.0 }; };

Stat bench_LU(const vector<double>& A, const vector<double>& f, size_t N, int reps)
{
    const double norm_x_star = sqrt(static_cast<double>(N));
    double t_sum = 0.0, delta = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto M = A; vector<size_t> piv; vector<double> x;
        auto t0 = hr_clock::now();
        lu_decompose(M, piv, N);
        lu_solve(M, piv, x, f, N);
        auto t1 = hr_clock::now();
        t_sum += seconds(t1 - t0).count();
        if (r == 0) {
            double diff2 = 0.0; for (double xi : x) diff2 += (xi - 1.0) * (xi - 1.0);
            delta = sqrt(diff2) / norm_x_star;
        }
    }
    return { t_sum / reps, delta };
}

Stat bench_QR(const vector<double>& A, const vector<double>& f, size_t N, int reps)
{
    const double norm_x_star = sqrt(static_cast<double>(N));
    double t_sum = 0.0, delta = 0.0;
    for (int r = 0; r < reps; ++r) {
        vector<double> x;
        auto t0 = hr_clock::now();
        qr_givens_solve(A, f, x, N);
        auto t1 = hr_clock::now();
        t_sum += seconds(t1 - t0).count();
        if (r == 0) {
            double diff2 = 0.0; for (double xi : x) diff2 += (xi - 1.0) * (xi - 1.0);
            delta = sqrt(diff2) / norm_x_star;
        }
    }
    return { t_sum / reps, delta };
}

void run_case(size_t N, int reps)
{
    vector<double> A(N * N), f(N);
    build_A(A, N);
    build_f(A, f, N);

    auto lu = bench_LU(A, f, N, reps);
    auto qr = bench_QR(A, f, N, reps);

    cout << "N = " << N << endl;
    cout << left << "\t" << "\t" << "Метод"
        << right << "\t" << "\t" << "время, c"
        << "\t" << "\t" << "б" << "\n";


    cout << left << "\t" << "\t" << "LU" << right
        << "\t" << "\t" << fixed << setprecision(8) << lu.avg_time
        << scientific << "\t" << "\t" << setprecision(8) << lu.delta << "\n";


    cout << left << "\t" << "\t" << "QR" << right
        << "\t" << "\t" << fixed << setprecision(8) << qr.avg_time
        << scientific << "\t" << "\t" << setprecision(8) << qr.delta << "\n" << "\n";
}

int main()
{
    setlocale(LC_ALL, "Rus");
    const int REPS = 100;
    for (size_t N : {250, 500, 1000})
        run_case(N, REPS);
    return 0;
}
