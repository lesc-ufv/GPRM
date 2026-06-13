#ifndef GPRM_CONFIG_HPP
#define GPRM_CONFIG_HPP

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include "src/hpp/enums/enum_operation.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class GPRMConfig {
private:
    int out_dim;
    int n_heads;
    std::unordered_map<std::string, int> vocab;
    bool use_mlp;
    double dropout;
    bool use_rms_norm;
    double negative_slope;
    int max_len;
    int lpe_dim;
    int feature_out_dim;
    int ff_dim;
    int n_transformer_layers;
    std::vector<int> gin_layers_list;
    int n_graph_embed_layers;
    bool use_transformer, use_gnn, use_mobility_info, use_coord_info;
    bool use_gin_emb, use_placement_order, norm_initial_emb, use_norm_before_pred;
    std::string activation_func;
    bool ignore_lpe;
    bool use_scale;

public:
    
    GPRMConfig()
    : out_dim(0), n_heads(0),
    use_mlp(true), dropout(0.0), use_rms_norm(true), negative_slope(0.0), max_len(0), lpe_dim(0), 
    feature_out_dim(0), ff_dim(0), n_transformer_layers(0), n_graph_embed_layers(0), 
    use_transformer(true), use_gnn(true), use_mobility_info(true), use_coord_info(true),use_gin_emb(true), 
    use_placement_order(true), norm_initial_emb(true), use_norm_before_pred(true), activation_func(""),ignore_lpe(false), use_scale(true){

        std::vector<std::string> vocab_list = {
        "Any", "add", "mult", "load", "output", 
        "store", "sub", "div", "shift", "compare"
    };

        int id = 0;
        for (const auto& word : vocab_list) {
            this->vocab[word] = id++;
        }
    }

    // int getPlacedId() const;
    // int getPlacementOptionId() const;
    // int getNotPlacedId() const ;
    int getOutDim() const;
    bool getUseScale() const {
        return use_scale;
    }
    void setUseScale(bool value) {
        use_scale = value;
    }
    void setOutDim(int value);
    
    bool getIgnoreLpe() const {
        return ignore_lpe;
    }
    
    void setIgnoreLpe(bool value) {
        ignore_lpe = value;
    }

    void set_config_from_json(const json& js) {
        lpe_dim               = js.at("lpe_dim").get<int>();
        feature_out_dim       = js.at("feat_out_dim").get<int>();
        out_dim               = js.at("out_dim").get<int>();
        ff_dim                = js.at("ff_out_dim").get<int>();
        gin_layers_list       = js.at("GIN_layers_list").get<std::vector<int>>();
        n_graph_embed_layers  = js.at("n_graph_embed_layers").get<int>();
        n_transformer_layers  = js.at("n_transformer_layers").get<int>();
        n_heads               = js.at("n_heads").get<int>();
        activation_func       = js.at("activation_func").get<std::string>();
        use_mlp               = js.at("use_mlp").get<bool>();
        dropout               = js.at("dropout").get<double>();
        use_rms_norm          = js.at("use_rms_norm").get<bool>();
        use_norm_before_pred  = js.at("use_norm_before_pred").get<bool>();
        negative_slope        = js.at("negative_slope").get<double>();
        max_len               = js.at("max_len").get<int>();
        norm_initial_emb      = js.at("norm_initial_emb").get<bool>();
        use_gin_emb           = js.at("use_gin_emb").get<bool>();
        use_mobility_info     = js.at("use_mobility_info").get<bool>();
        use_coord_info        = js.at("use_coord_info").get<bool>();
        use_placement_order   = js.at("use_placement_order").get<bool>();
        use_gnn               = js.at("use_gnn").get<bool>();
        use_transformer       = js.at("use_transformer").get<bool>();
        ignore_lpe            = js.at("ignore_lpe").get<bool>();
        use_scale             = js.at("use_scale").get<bool>();
    }
    
    
    int getNHeads() const;
    void setNHeads(int value);

    std::unordered_map<std::string, int> getVocab() const;

    int getVocabLen() const;

    // int getNotPlacementOptionId() const { return this->vocab.at("NOT_PLACEMENT_OPTION"); }
    bool getUseMobilityInfo() const {
        return this->use_mobility_info;
    };

    void setUseMobilityInfo(bool value) {
        this->use_mobility_info = value;
    };

    bool getUseCoordInfo() const {
        return this->use_coord_info;
    };

    void setUseCoordInfo(bool value) {
        this->use_coord_info = value;
    };

    bool getUseMlp() const;
    void setUseMlp(bool value);

    double getDropout() const;
    void setDropout(double value);

    bool getUseRmsNorm() const;
    void setUseRmsNorm(bool value);

    double getNegativeSlope() const;
    void setNegativeSlope(double value);

    int getMaxLen() const;
    void setMaxLen(int value);

    int getLpeDim() const;
    void setLpeDim(int value);

    int getFeatureOutDim() const;
    void setFeatureOutDim(int value);

    int getFfDim() const;
    void setFfDim(int value);

    int getNTransformerLayers() const;
    void setNTransformerLayers(int value);

    std::vector<int> getGinLayersList() const;
    void setGinLayersList(const std::vector<int>& value);

    int getNGraphEmbedLayers() const;
    void setNGraphEmbedLayers(int value);

    bool getUseTransformer() const;
    void setUseTransformer(bool value);

    bool getUseGnn() const;
    void setUseGnn(bool value);

    bool getUseGinEmb() const;
    void setUseGinEmb(bool value);
    
    bool getUsePlacementOrder() const{
        return this->use_placement_order;
    };
    void setUsePlacementOrder(bool value){
        this->use_placement_order = value;
    };

    bool getNormInitialEmb() const;
    void setNormInitialEmb(bool value);

    bool getUseNormBeforePred() const;
    void setUseNormBeforePred(bool value);

    std::string getActivationFunc() const;
    void setActivationFunc(const std::string& value);

    void print_ss(std::stringstream& ss) const;
};

#endif
