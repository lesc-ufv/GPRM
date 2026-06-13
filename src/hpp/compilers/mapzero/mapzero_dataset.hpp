#ifndef MAPZERO_DATASET
#define MAPZERO_DATASET
#include "src/hpp/datasets/mapping_dataset.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_state.hpp"

class MapZeroDataset : public MappingDataset<MapZeroState> {
public:
    using MappingDataset<MapZeroState>::MappingDataset;
};

#endif