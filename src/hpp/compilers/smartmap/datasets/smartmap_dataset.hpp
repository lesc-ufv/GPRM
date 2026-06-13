#ifndef SMARTMAP_DATASET
#define SMARTMAP_DATASET
#include "src/hpp/datasets/mapping_dataset.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"

class SmartMapDataset: public MappingDataset<SmartMapState>{
public: 
    using MappingDataset<SmartMapState>::MappingDataset;
};

#endif