#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <map>

class GraphSolver {
private:
    int repeat; 
    int revealed;
    void read_hypergraph(std::string filename);
    void read_labels(std::string filename);
    Eigen::SparseMatrix<double> create_laplacian(); 
    inline double fmax(double a, double b);
    inline double fmin(double a, double b);

public:
    int n; 
    int m; 
    std::string graph_name;
    Eigen::VectorXd degree; 
    std::vector<std::vector<int>> hypergraph; 
    std::vector<double> weights; 
    std::vector<std::vector<double>> hypergraph_node_weights; 
    std::vector<int> center_id; 
    bool has_hyperedge_centers; 
    int label_count;    
    std::vector<int> labels; 
    int early_stopping; 
    int verbose;        
    int preconditionerType; 

    GraphSolver(std::string graph_filename, std::string label_filename, std::string preconditioner, int verbose = 0);
    GraphSolver(int n_val, int m_val, Eigen::VectorXd degree_val, 
                std::vector<std::vector<int>> hypergraph_val, 
                int label_count_val = 0, std::vector<int> labels_val = std::vector<int>(), 
                int verbose_val = 0);

    Eigen::VectorXd infinity_subgradient(const Eigen::VectorXd& x_vec); // Single parameter
    Eigen::VectorXd diffusion(const Eigen::VectorXd& s_vec, int T, double lambda, double h_step, int schedule = 0);
    
    double compute_fx(const Eigen::VectorXd& x_vec, const Eigen::VectorXd& s_vec, double lambda, int t_fx_param = 1);
    double compute_error(const Eigen::VectorXd& x_vec_single_class_scores); 
    void run_diffusions(std::string graph_name_param, int repeats_param, int T_param, 
                        double lambda_param, double h_param, int minimum_revealed, 
                        int step_param, int maximum_revealed, int schedule_param = 0);
};