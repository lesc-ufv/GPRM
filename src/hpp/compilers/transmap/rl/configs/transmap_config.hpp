#ifndef TRANSMAP_CONFIG_HPP
#define TRANSMAP_CONFIG_HPP

#include <string>

class TransMapConfig {
private:
    int max_dfg_feat = 7;
    int max_adg_feat = 10;
    int lpe_dim;
    int E_dfg_node_dim;
    int E_dfg_edge_dim;
    int E_adg_node_dim;
    int E_adg_edge_dim;
    int out_dim_policy;
    int state_dim;
    int out_dim;
    int ff_dim;
    int n_heads;
    int nb_features = -1;
    int ortho_scaling;
    double dropout;
    int hidden_dim;
    std::string activation_fn;
    int num_mhd_layers;

public:
    TransMapConfig() = default;
    int getNumMhdLayers() const { return num_mhd_layers; }
    void setNumMhdLayers(int value) { num_mhd_layers = value; }
    int getMaxDfgFeat() const { return max_dfg_feat; }
    int getMaxAdgFeat() const { return max_adg_feat; }
    int getLpeDim() const { return lpe_dim; }
    int getEDfgNodeDim() const { return E_dfg_node_dim; }
    int getEDfgEdgeDim() const { return E_dfg_edge_dim; }
    int getEAdgNodeDim() const { return E_adg_node_dim; }
    int getEAdgEdgeDim() const { return E_adg_edge_dim; }
    int getOutDimPolicy() const { return out_dim_policy; }
    int getStateDim() const { return state_dim; }
    int getOutDim() const { return out_dim; }
    int getFfDim() const { return ff_dim; }
    int getNHeads() const { return n_heads; }
    int getNbFeatures() const { return nb_features; }
    int getOrthoScaling() const { return ortho_scaling; }
    double getDropout() const { return dropout; }
    std::string getActivationFn() const { return activation_fn; }
    int getHiddenDim() const {return hidden_dim;};

    void set_config_from_json(const json& js) {
        max_dfg_feat = js.at("max_dfg_feat").get<int>();
        max_adg_feat = js.at("max_adg_feat").get<int>();
        lpe_dim = js.at("lpe_dim").get<int>();
        E_dfg_node_dim = js.at("E_dfg_node_dim").get<int>();
        E_dfg_edge_dim = js.at("E_dfg_edge_dim").get<int>();
        E_adg_node_dim = js.at("E_adg_node_dim").get<int>();
        E_adg_edge_dim = js.at("E_adg_edge_dim").get<int>();
        out_dim_policy = js.at("out_dim_policy").get<int>();
        state_dim = js.at("state_dim").get<int>();
        out_dim = js.at("out_dim").get<int>();
        ff_dim = js.at("ff_dim").get<int>();
        n_heads = js.at("n_heads").get<int>();
        nb_features = js.at("nb_features").get<int>();
        hidden_dim = js.at("hidden_dim").get<int>();
        ortho_scaling = js.at("ortho_scaling").get<int>();
        dropout = js.at("dropout").get<double>();
        activation_fn = js.at("activation_fn").get<std::string>();
    }
    

    void setMaxDfgFeat(int value) { max_dfg_feat = value; }
    void setHiddenDim(int value) { hidden_dim = value; }
    void setMaxAdgFeat(int value) { max_adg_feat = value; }
    void setLpeDim(int value) { lpe_dim = value; }
    void setEDfgNodeDim(int value) { E_dfg_node_dim = value; }
    void setEDfgEdgeDim(int value) { E_dfg_edge_dim = value; }
    void setEAdgNodeDim(int value) { E_adg_node_dim = value; }
    void setEAdgEdgeDim(int value) { E_adg_edge_dim = value; }
    void setOutDimPolicy(int value) { out_dim_policy = value; }
    void setStateDim(int value) { state_dim = value; }
    void setOutDim(int value) { out_dim = value; }
    void setFfDim(int value) { ff_dim = value; }
    void setNHeads(int value) { n_heads = value; }
    void setNbFeatures(int value) { nb_features = value; }
    void setOrthoScaling(int value) { ortho_scaling = value; }
    void setDropout(double value) { dropout = value; }
    void setActivationFn(const std::string& value) { activation_fn = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nTransMapConfig:" << std::endl;
        ss << "\tMax DFG Feat: " << max_dfg_feat << std::endl;
        ss << "\tMax ADG Feat: " << max_adg_feat << std::endl;
        ss << "\tLPE Dim: " << lpe_dim << std::endl;
        ss << "\tE_DFG Node Dim: " << E_dfg_node_dim << std::endl;
        ss << "\tE_DFG Edge Dim: " << E_dfg_edge_dim << std::endl;
        ss << "\tE_ADG Node Dim: " << E_adg_node_dim << std::endl;
        ss << "\tE_ADG Edge Dim: " << E_adg_edge_dim << std::endl;
        ss << "\tOut Dim Policy: " << out_dim_policy << std::endl;
        ss << "\tState Dim: " << state_dim << std::endl;
        ss << "\tOut Dim: " << out_dim << std::endl;
        ss << "\tFeedforward Dim: " << ff_dim << std::endl;
        ss << "\tNumber of Heads: " << n_heads << std::endl;
        ss << "\tNumber of Features: " << nb_features << std::endl;
        ss << "\tHidden Dim: " << hidden_dim << std::endl;
        ss << "\tOrtho Scaling: " << ortho_scaling << std::endl;
        ss << "\tDropout: " << dropout << std::endl;
        ss << "\tActivation Function: " << activation_fn << std::endl;
    }
    
};

#endif
