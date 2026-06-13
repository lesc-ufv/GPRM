#ifndef TRANSMAP_BUFFER
#define TRANSMAP_BUFFER

template<typename StateT, typename ModelConfig>
void add_mapping_dtb(BaseBuffer<StateT,ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history){
    buffer.save_mapping_to_DTB(mapping_history);
}

template<typename StateT, typename ModelConfig>
MappingDataset<StateT> get_batch_dtb(BaseBuffer<StateT, ModelConfig>& buffer,
    std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
    std::string func_name){
    return buffer.get_batch_from_dtb(socket, server_k, batch_requester, func_name);
}

#endif