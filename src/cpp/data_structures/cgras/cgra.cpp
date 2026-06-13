#include "src/hpp/data_structures/cgras/cgra.hpp"

CGRA::CGRA(const CGRA& other)
    : II(other.II),
      dimensions(other.dimensions),
      pe_to_functionalities(other.pe_to_functionalities),
      interconnection_styles(other.interconnection_styles),
      PEs(other.PEs),
      connections(other.connections),
      pe_to_coord(other.pe_to_coord),
      coord_to_pe(other.coord_to_pe),
      PEs_to_routing(other.PEs_to_routing),
      free_connections(other.free_connections),
      in_vertices(other.in_vertices),
      out_vertices(other.out_vertices),
      graph(other.graph), 
      number_of_spatial_PEs(other.number_of_spatial_PEs),
      num_spatial_edges(other.num_spatial_edges) {
}

CGRA::CGRA( const unsigned int& II,
         const std::pair<int,int>& dimensions, 
         const std::unordered_map<int,std::unordered_set<EnumFunctionalities>>& pe_to_functionalities,
         const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles, bool calc_node_dists) : II(II), dimensions(dimensions), 
                                 pe_to_functionalities(pe_to_functionalities),
                                 interconnection_styles(interconnection_styles), num_spatial_edges(-1)
{   
    
    PEs = generate_PEs(dimensions, II);
    std::tie(pe_to_coord, coord_to_pe) = generate_PEs_coord_maps(dimensions, II);
    this->number_of_spatial_PEs = dimensions.first * dimensions.second;
    
    std::set<std::tuple<int, int>> edges_set;
    bool add_self_position = II > 1;

    for (const auto& interc_style : interconnection_styles) {
        for (const auto& PE : PEs) {
            auto adj_PEs = generate_connections_by_enum(interc_style, pe_to_coord, coord_to_pe, dimensions, PE, II, add_self_position);
            for (const auto& adj_PE : adj_PEs) {
                std::pair<int, int> edge = std::make_pair(PE, adj_PE);
                if (edges_set.find(edge) == edges_set.end()) {
                    edges_set.insert(edge);
                    connections.emplace_back(edge); 
                }
            }
        }
    }

    for (unsigned int i = 0; i < PEs.size(); i++) boost::add_vertex(graph);
    for (const auto& edge : connections) boost::add_edge(edge.first, edge.second, 1,graph);
    
    in_vertices = generate_cgra_in_vertices_dict(graph);
    out_vertices = generate_cgra_out_vertices_dict(graph);
    free_connections = out_vertices;

    this->edge_indices = create_edges_indices(this->connections);


    if (calc_node_dists) {
        auto g = this->graph;

        #ifdef DEBUG
        std::cout << "[DEBUG] Computing distance annotations using Dijkstra." << std::endl;
        #endif

        for (auto& source : this->PEs) {
            std::vector<int> distances(this->PEs.size(), std::numeric_limits<int>::max());
            auto distance_map = boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, g));
            dijkstra_shortest_paths(g, source, boost::distance_map(distance_map));

            #ifdef DEBUG
            std::cout << "[DEBUG] Distances from node " << source << ":" << std::endl;
            #endif

            for (size_t target = 0; target < distances.size(); ++target) {
                if (distances[target] != std::numeric_limits<int>::max() && distances[target] != 0) { 
                    node_to_out_dist_to_nodes[source][distances[target]].emplace_back(target);
                    node_to_in_dist_to_nodes[target][distances[target]].emplace_back(source);

                    #ifdef DEBUG
                    std::cout << "   - To node " << target << ": " << distances[target] << std::endl;
                    #endif
                }
            }
        }

        #ifdef DEBUG
        std::cout << "[DEBUG] Distance annotations computed." << std::endl;
        #endif
    }
    
    this->calc_IO_to_count();
    this->edge_to_id = create_edge_to_id(this->edge_indices);
}

CGRA::CGRA() : II(0), dimensions(0, 0),  pe_to_functionalities(), interconnection_styles() { }

// std::vector<std::pair<int, int>> CGRA::get_edges() const {
//     return this->connections;
// }

// std::vector<int> CGRA::get_vertices() const {
//     return this->PEs;
// }

int CGRA::total_PEs() const {
    return this->PEs.size();
}

const std::vector<std::pair<int, int>>& CGRA::get_edges_() const {
    return this->connections;
}

const std::vector<int>& CGRA::get_vertices_() const {
    return this->PEs;
}


// std::unordered_set<int> CGRA::get_in_vertices_by_id(const int& id) const {
//     return this->in_vertices.at(id);
// }

// std::unordered_set<int> CGRA::get_out_vertices_by_id(const int& id) const {
//     return this->out_vertices.at(id);
// }
const std::unordered_map<int,std::unordered_set<EnumFunctionalities>>& CGRA::get_pe_to_functionalities() const{
    return this->pe_to_functionalities;
}
const std::unordered_set<int>& CGRA::get_in_vertices_by_id_(const int& id) const {
    return this->in_vertices.at(id);
}

const std::unordered_set<int>& CGRA::get_out_vertices_by_id_(const int& id) const {
    return this->out_vertices.at(id);
}

const std::pair<int,int>& CGRA::get_cgra_dimensions_() const{
    return this->dimensions;
}


const std::unordered_map<int, std::tuple<int, int, int>> &CGRA::get_PE_to_coord_() const
{
    return this->pe_to_coord;
}

const std::unordered_map< std::tuple<int, int, int>, int> &CGRA::get_coord_to_PE_() const
{
    return this->coord_to_pe;
}
const std::tuple<int, int, int>& CGRA::get_coord_by_PE_(const int& PE){
    return this->pe_to_coord.at(PE);
}
const std::unordered_map<std::pair<int, int>, std::vector<int>> &CGRA::get_PEs_to_routing_() const
{
    return this->PEs_to_routing;
}

const std::unordered_map<int,std::unordered_set<int>>& CGRA::get_free_connections_() const {
    return this->free_connections;
}

std::unordered_map<int,std::unordered_set<int>> CGRA::get_free_connections() const {
    return this->free_connections;
}

void CGRA::remove_connections(const int &father_node, const int &child_node)
{
    this->free_connections[father_node].erase(child_node);
}

const std::unordered_map<int, std::unordered_set<int>>& CGRA::get_out_vertices_() const {
    return this->out_vertices;
}

const std::unordered_map<int, std::unordered_set<int>>& CGRA::get_in_vertices_() const {
    return this->in_vertices;
}

std::pair<int,int> CGRA::get_pe_pos(const int& PE) const {
    int PE_pos0, PE_pos1;
    auto PE_coord = this->pe_to_coord.at(PE);
    PE_pos0 = std::get<1>(PE_coord);
    PE_pos1 = std::get<2>(PE_coord);
    return std::make_pair(PE_pos0, PE_pos1);
}

const int& CGRA::number_of_spatial_PEs_() const {
    return this->number_of_spatial_PEs;
}

void CGRA::add_routing(const std::pair<int, int> tgt_routing_PEs, std::vector<int> routing_path)
{
    this->PEs_to_routing[tgt_routing_PEs] = routing_path;
}
int CGRA::get_II() const
{
    return this->II;
}


void CGRA::print() const {
    std::cout << "=== CGRA ===\n";

    std::cout << "Initiation Interval (II): " << this->II << "\n";
    std::cout << "Dimensions: (" << dimensions.first << ", " << dimensions.second << ")\n";
    std::cout << "Number of Spatial PEs: " << this->number_of_spatial_PEs << "\n";

    std::cout << "\nProcessing Elements (PEs):\n";
    print_vector(PEs);

    std::cout << "\nPE Functionalities:\n";
    for (const auto& [pe, functionalities] : this->pe_to_functionalities) {
        std::cout << "PE " << pe << " -> Functionalities: ";
        for (const auto& func : functionalities) {
            std::cout << static_cast<int>(func) << " ";  
        }
        std::cout << "\n";
    }

    std::cout << "\nInterconnection Styles:\n";
    for (const auto& style : interconnection_styles) {
        std::cout << static_cast<int>(style) << " ";  
    }
    std::cout << "\n";

    std::cout << "\nConnections (Edges):\n";
    print_pair_vector(connections);

    std::cout << "\nPE to coordinate Mapping:\n";
    for (const auto& [pe, coord] : pe_to_coord) {
        std::cout << "PE " << pe << " -> (" 
                  << std::get<0>(coord) << ", "
                  << std::get<1>(coord) << ", "
                  << std::get<2>(coord) << ")\n";
    }

    std::cout << "\nCoordinate to PE Mapping:\n";
    for (const auto& [coord, pe] : coord_to_pe) {
        std::cout << "Coordinate (" 
                  << std::get<0>(coord) << ", "
                  << std::get<1>(coord) << ", "
                  << std::get<2>(coord) << ") -> PE " << pe << "\n";
    }

    std::cout << "\nPEs to Routing Paths:\n";
    for (const auto& [pes, route] : PEs_to_routing) {
        std::cout << "PEs (" << pes.first << ", " << pes.second << ") -> Route: ";
        print_vector(route);
    }

    std::cout << "\nFree Connections:\n";
    print_element_key_list_value(free_connections);

    std::cout << "\nIn Vertices:\n";
    print_element_key_list_value(in_vertices);
    std::cout << "\nOut Vertices:\n";
    print_element_key_list_value(out_vertices);
    
    std::cout << "=== End of CGRA ===\n";
}

int CGRA::get_n_rows(){
    return this->dimensions.first;
}

int CGRA::get_n_cols(){
    return this->dimensions.second;
}

void CGRA::add_self_edges() {
    for (const auto& PE : this->PEs) {
        connections.emplace_back(std::make_pair(PE, PE));
    }
}

const std::unordered_map<std::pair<int,int>,int>& CGRA::get_edge_to_id(){
    return this->edge_to_id;
}

std::string CGRA::get_supported_operations_string_by_node(const int& node) const {
    return "Any";
}

int CGRA::get_num_edges() const{
    return this->get_edges_().size();
}

const std::unordered_set<int>& CGRA::get_in_vertices_by_node_id(const int& node) const {
    return this->in_vertices.at(node);
}

const std::unordered_set<int>& CGRA::get_out_vertices_by_node_id(const int& node) const {
    return this->out_vertices.at(node);
}


const std::vector<std::vector<int>>& CGRA::get_edge_indices() const{
    return this->edge_indices;
}

const std::unordered_map<int, std::unordered_map<int,std::vector<int>>>& CGRA::get_node_to_in_dist_to_nodes() const{
    return this->node_to_in_dist_to_nodes;
}

const std::unordered_map<int, std::unordered_map<int,std::vector<int>>>& CGRA::get_node_to_out_dist_to_nodes() const{
    return this->node_to_out_dist_to_nodes;
}