#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"


// int GPRMConfig::getPlacedId() const { return this->vocab.at("PLACED"); }
// int GPRMConfig::getPlacementOptionId() const { return this->vocab.at("PLACEMENT_OPTION"); }

// int GPRMConfig::getNotPlacedId() const { return this->vocab.at("NOT_PLACED"); }

int GPRMConfig::getOutDim() const { return out_dim; }
void GPRMConfig::setOutDim(int value) { out_dim = value; }

int GPRMConfig::getNHeads() const { return n_heads; }
void GPRMConfig::setNHeads(int value) { n_heads = value; }

std::unordered_map<std::string, int> GPRMConfig::getVocab() const { return vocab; }

int GPRMConfig::getVocabLen() const { return vocab.size(); }


bool GPRMConfig::getUseMlp() const { return use_mlp; }
void GPRMConfig::setUseMlp(bool value) { use_mlp = value; }

double GPRMConfig::getDropout() const { return dropout; }
void GPRMConfig::setDropout(double value) { dropout = value; }

bool GPRMConfig::getUseRmsNorm() const { return use_rms_norm; }
void GPRMConfig::setUseRmsNorm(bool value) { use_rms_norm = value; }

double GPRMConfig::getNegativeSlope() const { return negative_slope; }
void GPRMConfig::setNegativeSlope(double value) { negative_slope = value; }

int GPRMConfig::getMaxLen() const { return max_len; }
void GPRMConfig::setMaxLen(int value) { max_len = value; }

int GPRMConfig::getLpeDim() const { return lpe_dim; }
void GPRMConfig::setLpeDim(int value) { lpe_dim = value; }

int GPRMConfig::getFeatureOutDim() const { return feature_out_dim; }
void GPRMConfig::setFeatureOutDim(int value) { feature_out_dim = value; }

int GPRMConfig::getFfDim() const { return ff_dim; }
void GPRMConfig::setFfDim(int value) { ff_dim = value; }

int GPRMConfig::getNTransformerLayers() const { return n_transformer_layers; }
void GPRMConfig::setNTransformerLayers(int value) { n_transformer_layers = value; }

std::vector<int> GPRMConfig::getGinLayersList() const { return gin_layers_list; }
void GPRMConfig::setGinLayersList(const std::vector<int>& value) { gin_layers_list = value; }

int GPRMConfig::getNGraphEmbedLayers() const { return n_graph_embed_layers; }
void GPRMConfig::setNGraphEmbedLayers(int value) { n_graph_embed_layers = value; }

bool GPRMConfig::getUseTransformer() const { return use_transformer; }
void GPRMConfig::setUseTransformer(bool value) { use_transformer = value; }

bool GPRMConfig::getUseGnn() const { return use_gnn; }
void GPRMConfig::setUseGnn(bool value) { use_gnn = value; }


bool GPRMConfig::getUseGinEmb() const { return use_gin_emb; }
void GPRMConfig::setUseGinEmb(bool value) { use_gin_emb = value; }

bool GPRMConfig::getNormInitialEmb() const { return norm_initial_emb; }
void GPRMConfig::setNormInitialEmb(bool value) { norm_initial_emb = value; }

bool GPRMConfig::getUseNormBeforePred() const { return use_norm_before_pred; }
void GPRMConfig::setUseNormBeforePred(bool value) { use_norm_before_pred = value; }

std::string GPRMConfig::getActivationFunc() const { return activation_func; }
void GPRMConfig::setActivationFunc(const std::string& value) { activation_func = value; }

void GPRMConfig::print_ss(std::stringstream& ss) const {
    ss << "\nGPRMConfig:" << std::endl;
    ss << "\tOut Dim: " << out_dim << std::endl;
    ss << "\tNum Heads: " << n_heads << std::endl;
    ss << "\tVocab Length: " << getVocabLen() << std::endl;
    ss << "\tUse MLP: " << (use_mlp ? "true" : "false") << std::endl;
    ss << "\tDropout: " << dropout << std::endl;
    ss << "\tUse RMS Norm: " << (use_rms_norm ? "true" : "false") << std::endl;
    ss << "\tNegative Slope: " << negative_slope << std::endl;
    ss << "\tMax Length: " << max_len << std::endl;
    ss << "\tLPE Dim: " << lpe_dim << std::endl;
    ss << "\tFeature Out Dim: " << feature_out_dim << std::endl;
    ss << "\tFF Dim: " << ff_dim << std::endl;
    ss << "\tNum Transformer Layers: " << n_transformer_layers << std::endl;
    ss << "\tNum Graph Embed Layers: " << n_graph_embed_layers << std::endl;
    ss << "\tUse Transformer: " << (use_transformer ? "true" : "false") << std::endl;
    ss << "\tUse GNN: " << (use_gnn ? "true" : "false") << std::endl;
    ss << "\tUse GIN Emb: " << (use_gin_emb ? "true" : "false") << std::endl;
    ss << "\tNormalize Initial Embedding: " << (norm_initial_emb ? "true" : "false") << std::endl;
    ss << "\tUse Norm Before Prediction: " << (use_norm_before_pred ? "true" : "false") << std::endl;
    ss << "\tUse Placement Order: " << (use_placement_order ? "true" : "false") << std::endl;
    ss << "\tActivation Function: " << activation_func << std::endl;
}
