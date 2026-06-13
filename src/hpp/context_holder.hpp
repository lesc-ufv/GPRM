#ifndef CONTEXT_HOLDER_HPP
#define CONTEXT_HOLDER_HPP

class ZmqContextHolder {
    public:
        static zmq::context_t& instance() {
            static zmq::context_t ctx(1);
            return ctx;
        }
};
    
#endif