// diffusion.cpp
#include <cmath>
#include <algorithm> 
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <iterator>
#include <chrono>
#include <ctime>   
#include <cassert> 
#include <limits>  
#include <random>  
#include <iomanip> 
#include <numeric> 

#include <Eigen/Dense> 
#include <Eigen/Sparse> 

#include "diffusion.h"

namespace { 
    // Corrected RNG names
    static std::random_device g_rd_static_cpp_diffusion; 
    static std::mt19937 g_rng_static_cpp_diffusion(g_rd_static_cpp_diffusion()); 
}


void GraphSolver::read_hypergraph(std::string filename) { 
    this->hypergraph.clear(); this->weights.clear(); this->hypergraph_node_weights.clear(); this->center_id.clear();
    int fmt_read; std::string line; std::ifstream input_file; input_file.open(filename);
    if (!input_file.is_open()) { std::cerr << "ERROR: Failed to open hypergraph file: " << filename << std::endl; exit(1); }
    if (!(input_file >> this->m >> this->n >> fmt_read)) { std::cerr << "ERROR: Failed to read hypergraph header from: " << filename << std::endl; input_file.close(); exit(1); }
    getline(input_file, line);
    bool has_edge_weights = fmt_read % 10; bool has_node_weights_from_fmt = (fmt_read / 10) % 10;
    bool hyperedge_has_node_weights = (fmt_read / 100) % 10; this->has_hyperedge_centers = (fmt_read / 1000) % 10;
    if (this->n <= 0 || this->m < 0) { std::cerr << "ERROR: Invalid n=" << this->n << " or m=" << this->m << std::endl; input_file.close(); exit(1); }
    this->hypergraph.reserve(this->m); this->weights.reserve(this->m); this->hypergraph_node_weights.reserve(this->m); this->center_id.reserve(this->m);
    for(int i = 0; i < this->m; i++) {
        std::vector<int> hyperedge; std::vector<double> hyperedge_node_weights_local; int node_val; double d_value;
        if (!getline(input_file, line)) { std::cerr << "ERROR: Failed to read line for hyperedge " << i << std::endl; input_file.close(); exit(1); }
        std::istringstream iss(line);
        if(has_edge_weights) { if (!(iss >> d_value)) { std::cerr << "ERROR: Reading edge weight " << i << std::endl; exit(1); } this->weights.push_back(d_value); } else { this->weights.push_back(1.0); }
        if(this->has_hyperedge_centers) { if (!(iss >> node_val)) { std::cerr << "ERROR: Reading center " << i << std::endl; exit(1); } if (node_val -1 < 0 || node_val -1 >= this->n) { std::cerr << "ERROR: Center Node ID " << node_val << " OOB " << i << std::endl; exit(1); } this->center_id.push_back(node_val - 1); }
        while(iss >> node_val) { if (node_val - 1 < 0 || node_val - 1 >= this->n) { std::cerr << "ERROR: Node ID " << node_val << " OOB " << i << std::endl; exit(1); } hyperedge.push_back(node_val - 1); if(hyperedge_has_node_weights) { if (!(iss >> d_value)) { std::cerr << "ERROR: Reading node weight in h " << i << std::endl; exit(1); } hyperedge_node_weights_local.push_back(d_value); } }
        this->hypergraph.push_back(hyperedge); this->hypergraph_node_weights.push_back(hyperedge_node_weights_local);
    }
    this->degree = Eigen::VectorXd(this->n);
    if(has_node_weights_from_fmt) { for(int i = 0; i < this->n; i++) { double d_val_from_file; if (!(input_file >> d_val_from_file)) { for (int k_fill = i; k_fill < this->n; ++k_fill) this->degree(k_fill) = 1.0; break; } this->degree(i) = d_val_from_file; }
    } else { this->degree.setOnes(); } input_file.close();
}
void GraphSolver::read_labels(std::string filename) { 
    if (this->n == 0 && filename.empty()) { this->label_count = 0; this->labels.clear(); return; } if (filename.empty()){ this->label_count = 0; this->labels.assign(this->n > 0 ? this->n : 0, 0); return; }
    std::ifstream label_file; label_file.open(filename); if (!label_file.is_open()) { this->label_count = 0; this->labels.assign(this->n > 0 ? this->n : 0, 0); return; }
    this->labels.clear(); if (this->n > 0) this->labels.reserve(this->n); std::map<int, int> unique_label_map; int next_unique_id = 0; std::string header_line; if (label_file.good() && label_file.peek() != EOF) { getline(label_file, header_line); } 
    for(int i = 0; i < this->n; i++) { int original_label; if (!(label_file >> original_label)) { if (i < this->n) {std::cerr << "WARN: Label read short.\n";} for (int k_fill = i; k_fill < this->n; ++k_fill) this->labels.push_back(0); break; } if (unique_label_map.find(original_label) == unique_label_map.end()) { unique_label_map[original_label] = next_unique_id++; } this->labels.push_back(unique_label_map[original_label]); }
    label_file.close(); this->label_count = unique_label_map.size(); if (this->labels.size() != this->n && this->n > 0) { this->labels.resize(this->n, 0); }
}
inline double GraphSolver::fmax(double a, double b) { return a > b ? a : b; }
inline double GraphSolver::fmin(double a, double b) { return a < b ? a : b; }

Eigen::SparseMatrix<double> GraphSolver::create_laplacian() { 
    if (this->n <= 0 || this->m < 0) return Eigen::SparseMatrix<double>(); 
    Eigen::SparseMatrix<double> laplacian(this->n + this->m, this->n + this->m); 
    std::vector<Eigen::Triplet<double>> tripletList; 
    tripletList.reserve(this->n * 2 + this->m * 10); 
    for(int j_idx = 0; j_idx < this->m; j_idx++) { 
        if (j_idx >= this->hypergraph.size()) { std::cerr << "ERROR: create_laplacian: j_idx OOB\n"; exit(1); } 
        const auto& current_hyperedge = this->hypergraph[j_idx]; 
        size_t h_size = current_hyperedge.size(); 
        if (h_size > 0) { 
            tripletList.push_back(Eigen::Triplet<double>(this->n + j_idx, this->n + j_idx, static_cast<double>(h_size))); 
            for(int node_in_edge : current_hyperedge) { 
                if (node_in_edge < 0 || node_in_edge >= this->n) { std::cerr << "ERROR: create_laplacian: node_in_edge OOB\n"; exit(1); } 
                tripletList.push_back(Eigen::Triplet<double>(node_in_edge, node_in_edge, 1.0)); 
                tripletList.push_back(Eigen::Triplet<double>(node_in_edge, this->n + j_idx, -1.0)); 
                tripletList.push_back(Eigen::Triplet<double>(this->n + j_idx, node_in_edge, -1.0)); 
            } 
        } 
    }
    laplacian.setFromTriplets(tripletList.begin(), tripletList.end()); 
    return laplacian;
}
GraphSolver::GraphSolver(std::string gf, std::string lf, std::string pn, int vv): graph_name(gf), early_stopping(-1), verbose(vv), has_hyperedge_centers(false), n(0), m(0), preconditionerType(0), label_count(0) { read_hypergraph(gf); if(this->n == 0){ exit(1); } if(!lf.empty()) { read_labels(lf); } else { this->labels.assign(this->n, 0); this->label_count = (this->n > 0) ? 1:0; } if (pn.compare("degree") == 0) { this->preconditionerType = 0; } else { exit(1); } }
GraphSolver::GraphSolver(int n_v, int m_v, Eigen::VectorXd deg_v, std::vector<std::vector<int>> h_v, int lc_v, std::vector<int> ls_v, int vv_v): n(n_v), m(m_v), graph_name(""), degree(deg_v), hypergraph(h_v), label_count(lc_v), early_stopping(-1), verbose(vv_v), preconditionerType(0), has_hyperedge_centers(false) { this->labels = ls_v; }

// --- infinity_subgradient (NEW IMPLEMENTATION from previous response) ---
Eigen::VectorXd GraphSolver::infinity_subgradient(const Eigen::VectorXd& x_vec) {
    Eigen::VectorXd gradient(this->n);
    gradient.setZero();
    for (int j_idx = 0; j_idx < this->m; ++j_idx) { 
        const auto& current_hyperedge = this->hypergraph[j_idx];
        if (current_hyperedge.empty()) { continue; }
        double x_min_in_h = x_vec(current_hyperedge[0]); 
        double x_max_in_h = x_vec(current_hyperedge[0]);
        for (size_t i = 1; i < current_hyperedge.size(); ++i) { 
            int node_id = current_hyperedge[i];
            if (node_id < 0 || node_id >= this->n) { exit(1); }
            double val = x_vec(node_id);
            if (val < x_min_in_h) x_min_in_h = val;
            if (val > x_max_in_h) x_max_in_h = val;
        }
        if (std::abs(x_max_in_h - x_min_in_h) < 1e-12) { continue; }
        std::vector<int> argmax_nodes; std::vector<int> argmin_nodes;
        argmax_nodes.reserve(current_hyperedge.size()); argmin_nodes.reserve(current_hyperedge.size());
        for (int node_id : current_hyperedge) {
            if (std::abs(x_vec(node_id) - x_max_in_h) < 1e-9) { argmax_nodes.push_back(node_id); }
            if (std::abs(x_vec(node_id) - x_min_in_h) < 1e-9) { argmin_nodes.push_back(node_id); }
        }
        double hyperedge_weight = 1.0; 
        if (j_idx < this->weights.size()){ hyperedge_weight = this->weights[j_idx]; }
        double diff_term_scaled = 0.5 * hyperedge_weight * (x_max_in_h - x_min_in_h);
        if (!argmax_nodes.empty()) {
            double grad_val_for_max_nodes = diff_term_scaled / static_cast<double>(argmax_nodes.size());
            for (int node_id : argmax_nodes) { gradient(node_id) += grad_val_for_max_nodes; }
        }
        if (!argmin_nodes.empty()) {
            double grad_val_for_min_nodes = -diff_term_scaled / static_cast<double>(argmin_nodes.size());
            for (int node_id : argmin_nodes) { gradient(node_id) += grad_val_for_min_nodes; }
        }
    }
    return gradient;
}

// --- diffusion (with detailed logging, calls single-arg infinity_subgradient) ---
Eigen::VectorXd GraphSolver::diffusion(const Eigen::VectorXd& s_vec, int T, double lambda, double h_step, int schedule) {
    // ... (Full implementation as in my previous "complete" reply, with all diagnostic prints)
    if (this->n == 0) { return Eigen::VectorXd(); }
    if (s_vec.size() != this->n) { return Eigen::VectorXd(this->n).setZero(); }
    const auto start_time_diffusion{std::chrono::steady_clock::now()};
    int function_stopping = early_stopping; int best_t_iter = 0; 
    double step = h_step; double best_fx_val = std::numeric_limits<double>::infinity(); 
    int best_fx_unchanged_count = 0;
    Eigen::VectorXd x(this->n); x.setZero(); Eigen::VectorXd dx(this->n); dx.setZero();
    Eigen::VectorXd solution_accumulator(this->n); solution_accumulator.setZero();
    Eigen::VectorXd best_solution_vec(this->n); best_solution_vec.setZero();

    for(int t_loop = 0; t_loop < T; ++t_loop) {
        Eigen::VectorXd grad_from_inf_subgrad = infinity_subgradient(x); // SINGLE ARGUMENT CALL
        Eigen::VectorXd current_gradient = grad_from_inf_subgrad; 
        for(int i_idx = 0; i_idx < this->n; ++i_idx) {
            if (i_idx >= this->degree.size() || i_idx >= x.size() || i_idx >= s_vec.size()){ exit(1); }
            current_gradient(i_idx) += lambda * this->degree(i_idx) * x(i_idx) - s_vec(i_idx);
        }
        if (t_loop == 0) { 
            std::cerr << "  [C++ Diff Iter 0] s_vec norm: " << s_vec.norm(); if(s_vec.size()>0) std::cerr << ", s_vec[:5]: " << s_vec.head(std::min(5,(int)s_vec.size())).transpose(); std::cerr << std::endl;
            std::cerr << "  [C++ Diff Iter 0] grad_from_inf_subgrad norm: " << grad_from_inf_subgrad.norm() << std::endl;
            std::cerr << "  [C++ Diff Iter 0] current_gradient total norm: " << current_gradient.norm(); if(current_gradient.size()>0) std::cerr << ", current_gradient[:5]: " << current_gradient.head(std::min(5,(int)current_gradient.size())).transpose(); std::cerr << std::endl;
        }
        switch(this->preconditionerType) { 
            case 0: for(int i_idx = 0; i_idx < this->n; ++i_idx) { if (std::abs(this->degree(i_idx)) < 1e-12) { dx(i_idx) = current_gradient(i_idx) * 1e12; if(std::abs(current_gradient(i_idx)) > 1e-9) std::cerr << "WARNING: Zero deg node " << i_idx << " grad " << current_gradient(i_idx) << std::endl;} else { dx(i_idx) = current_gradient(i_idx) / this->degree(i_idx); }} break;
            default: dx = current_gradient; break;
        }
        if (schedule % 2 == 1 && t_loop > 0) { step = h_step / std::sqrt(static_cast<double>(t_loop + 1));} else { step = h_step; }
        x -= step * dx; solution_accumulator += x;
        if (t_loop < 5 || (t_loop + 1) % 100 == 0 || t_loop == T - 1) { 
            std::cerr << "  [C++ Diff Iter " << t_loop + 1 << "] x norm: " << x.norm() << ", x min: " << (x.size()>0?x.minCoeff():0) << ", x max: " << (x.size()>0?x.maxCoeff():0) << ", x mean: " << (x.size()>0?x.mean():0) << std::endl;
            std::cerr << "  [C++ Diff Iter " << t_loop + 1 << "] current_gradient norm: " << current_gradient.norm() << std::endl;
            std::cerr << "  [C++ Diff Iter " << t_loop + 1 << "] dx norm: " << dx.norm() << ", step: " << step << std::endl;
            int x_nnz = 0; for (int i=0; i<x.size(); ++i) if(std::abs(x(i)) > 1e-9) x_nnz++;
            std::cerr << "  [C++ Diff Iter " << t_loop + 1 << "] x non-zeros: " << x_nnz << "/" << this->n << std::endl;
             if (x.size() > 0 && x_nnz > 0 && x_nnz < 30 && t_loop > 0 && x_nnz != this->n) { std::cerr << "  [C++ Diff Iter " << t_loop + 1 << "] x values (first few non-zeros): "; int pc=0; for(int i=0;i<x.size()&&pc<15;++i){if(std::abs(x(i))>1e-9){std::cerr<<"idx"<<i<<":"<<x(i)<<" ";pc++;}} std::cerr<<std::endl;}
            std::cerr << "  ------------------------------------" << std::endl;
        }
        if (t_loop + 1 >= 1) { 
            Eigen::VectorXd avg_sol = solution_accumulator / static_cast<double>(t_loop + 1); double cur_sol_fx = this->compute_fx(avg_sol, s_vec, lambda, 1);
            if(cur_sol_fx < best_fx_val) { best_fx_val=cur_sol_fx; best_solution_vec=avg_sol; best_fx_unchanged_count=0; best_t_iter=t_loop+1; } else { best_fx_unchanged_count++; }
            if((function_stopping > 0) && (best_fx_unchanged_count > function_stopping)) { if((schedule / 2) % 2) {best_fx_unchanged_count=0; function_stopping=static_cast<int>(function_stopping*sqrt(2.0)); if(step<1e-5 && h_step<1e-5) break; h_step/=sqrt(2.0);} else break; }
        }
    }
    if (T > 0) { Eigen::VectorXd final_avg = solution_accumulator / static_cast<double>(T); if ( (best_t_iter == 0 && T > 0) || (T == best_t_iter && T > 0) || (best_fx_val > this->compute_fx(final_avg, s_vec, lambda, 1) ) ) { best_solution_vec = final_avg; } }
    return best_solution_vec;
}

// ... (compute_fx, compute_error as in previous complete version) ...
double GraphSolver::compute_fx(const Eigen::VectorXd& x_vec, const Eigen::VectorXd& s_vec, double lambda, int t_fx_param_unused ) { 
    double fx = 0; for(int j_idx = 0; j_idx < this->m; ++j_idx) { if (j_idx >= this->hypergraph.size() || this->hypergraph[j_idx].empty()) { continue; } double ymin = std::numeric_limits<double>::infinity(); double ymax = -std::numeric_limits<double>::infinity(); const auto& ch = this->hypergraph[j_idx]; for(int nid : ch) { if (nid < 0 || nid >= x_vec.size()) { exit(1); } ymin = fmin(ymin, x_vec(nid)); ymax = fmax(ymax, x_vec(nid)); } if (!std::isinf(ymax) && !std::isinf(ymin) && std::abs(ymax-ymin) > 1e-12) { fx += 0.5 * this->weights[j_idx] * (ymax - ymin) * (ymax - ymin); } }
    double term_L2_D = 0; double term_inner_prod = 0; for(int i_idx = 0; i_idx < this->n; ++i_idx) { if (i_idx >= x_vec.size() || i_idx >= this->degree.size() || i_idx >= s_vec.size()) { exit(1); } term_L2_D += this->degree(i_idx) * x_vec(i_idx) * x_vec(i_idx); term_inner_prod += s_vec(i_idx) * x_vec(i_idx); }
    fx += (lambda / 2.0) * term_L2_D; fx -= term_inner_prod; return fx;
}
double GraphSolver::compute_error(const Eigen::VectorXd& x_vec_single_class_scores) { return -1.0; }

// Corrected run_diffusions to use the corrected RNG name
void GraphSolver::run_diffusions(std::string gnp, int rp, int Tp, double lp, double hp, int mr, int sp, int xr, int schedp) { 
    std::cerr << "WARNING: run_diffusions called." << std::endl; 
    if (this->label_count <= 0 || this->n == 0) return; 
    Eigen::MatrixXd sf(this->label_count, this->n); 
    std::vector<int> oa(this->n); 
    std::iota(oa.begin(), oa.end(), 0); 
    for(this->repeat = 0; this->repeat < rp; ++this->repeat) { 
        sf.setZero(); 
        std::shuffle(oa.begin(), oa.end(), g_rng_static_cpp_diffusion); // Use corrected RNG name
        for(this->revealed = mr; this->revealed <= xr; this->revealed += sp) { 
            for(int r_idx = 0; r_idx < this->label_count; ++r_idx) { 
                for(int i_idx = 0; i_idx < this->revealed; ++i_idx) { 
                    int node = oa[i_idx]; 
                    if (node < 0 || node >= this->labels.size() || node >= this->n || r_idx < 0 || r_idx >= this->label_count) exit(1); 
                    sf(r_idx, node) = lp * ( (this->labels[node] == r_idx) ? 1.0 : -1.0 ); 
                } 
            } 
            Eigen::MatrixXd sol_mat(this->label_count, this->n); 
            for(int c=0; c < this->label_count; ++c) { 
                sol_mat.row(c) = this->diffusion(Eigen::VectorXd(sf.row(c)), Tp, lp, hp, schedp); 
            } 
            std::cout << gnp << ",C++(run_diff)," << this->repeat << "," << this->revealed << "," << lp << ","
                      << 0.0 << "," << "N/A_err" << "," << "N/A_fx" << "," << hp << std::endl; 
        } 
    } 
}