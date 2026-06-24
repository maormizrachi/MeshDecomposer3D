#ifndef HILBERT_RECTANGULAR_TREE_3D_HPP
#define HILBERT_RECTANGULAR_TREE_3D_HPP

#include "../HilbertConvertor3D.hpp"
#ifdef RICH_MPI

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>
#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <mpi.h>
#include <spatial_ds/utils/geometry.hpp>
#include "HilbertRectangularConvertor3D.hpp"
#include "../../error.hpp"

#define DEFAULT_RANKS_IN_LEAVES 4
#define UNDEFINED_OWNER -1

#ifndef EPSILON
#define EPSILON 1e-12
#endif

template<typename PointT, int max_leaf_ranks = DEFAULT_RANKS_IN_LEAVES>
class HilbertRectangularTree3DNode
{
public:
    explicit HilbertRectangularTree3DNode(HilbertRectangularTree3DNode *parent) : parent(parent) {}

    BoundingBox<PointT> boundingBox;
    hilbert_index_t d_start, d_end;
    size_t num_points;
    std::vector<HilbertRectangularTree3DNode *> children;
    bool is_leaf;
    boost::container::small_vector<int, max_leaf_ranks> owners;
    HilbertRectangularTree3DNode *parent;
};

template<typename PointT, int max_leaf_ranks = DEFAULT_RANKS_IN_LEAVES>
class HilbertRectangularTree3D
{
public:
    using Node = HilbertRectangularTree3DNode<PointT, max_leaf_ranks>;
    using coord_type = typename PointT::coord_type;

private:
    using RanksSet = boost::container::flat_set<int>;

    Node *root;
    mutable std::vector<const Node *> nodes_stack;
    MPI_Comm comm;
    int rank, size;
    size_t depth;
    const std::shared_ptr<HilbertConvertor3D<PointT>> convertor;

    void buildTreeHelper(Node *currentNode,
                         const typename HilbertRectangularConvertor3D<PointT>::RecursionArguments &current_args,
                         size_t currentDepth,
                         hilbert_index_t &current_d,
                         const std::shared_ptr<HilbertRectangularConvertor3D<PointT>> &rectConvertor,
                         const std::vector<hilbert_index_t> &responsibilityRange);

    void buildTree(const std::shared_ptr<HilbertRectangularConvertor3D<PointT>> &rectConvertor,
                   const std::vector<hilbert_index_t> &responsibilityRange);

    void printHelper(const Node *node, int tabs = 0) const;

public:
    HilbertRectangularTree3D(std::shared_ptr<HilbertRectangularConvertor3D<PointT>> convertor,
                             const std::vector<hilbert_index_t> &responsibilityRange,
                             const MPI_Comm &comm = MPI_COMM_WORLD)
        : comm(comm), convertor(convertor)
    {
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Comm_size(this->comm, &this->size);
        this->depth = 0;
        this->buildTree(convertor, responsibilityRange);
    }

    ~HilbertRectangularTree3D();

    template<typename U>
    RanksSet getIntersectingRanks(const Sphere<U> &sphere) const;

    template<typename U>
    inline RanksSet getIntersectingRanks(const U &point, typename U::coord_type radius) const
    {
        return this->getIntersectingRanks(Sphere<U>(point, radius));
    }

    template<typename U>
    std::vector<std::pair<coord_type, coord_type>> getClosestFurthestPointsByRanks(const U &point) const;

    void print() const { this->printHelper(this->root); }

    std::vector<BoundingBox<PointT>> getRankBoundingBoxes(int _rank) const;

    inline std::vector<BoundingBox<PointT>> getMyBoundingBoxes() const
    {
        return this->getRankBoundingBoxes(this->rank);
    }

    std::vector<std::vector<BoundingBox<PointT>>> getBoundingBoxesOfRanks(void) const;

    std::vector<const Node *> getValuesIf(const std::function<bool(const Node *)> ifOpenFunction,
                                          const std::function<bool(const Node *)> &ifAddValueFunction) const;

    inline size_t getDepth() const { return this->depth; }

    std::vector<int> getOwners(const PointT &point) const;
};

template<typename PointT, int max_ranks_per_leaf>
HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::~HilbertRectangularTree3D()
{
    this->nodes_stack.push_back(this->root);
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(!node->is_leaf)
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child);
            }
        }

        delete node;
    }

    this->root = nullptr;
}

template<typename PointT, int max_ranks_per_leaf>
void HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::printHelper(const Node *node, int tabs) const
{
    if(node == nullptr)
    {
        return;
    }
    for(int i = 0; i < tabs; i++) std::cout << "\t";
    std::cout << "LL = " << node->boundingBox.getLL() << ", UR = " << node->boundingBox.getUR() << ", d: " << node->d_start << " - " << node->d_end << " (points: " << node->num_points << "). ";

    size_t numOwners = node->owners.size();

    if(numOwners == 0)
    {
        std::cout << "No explicit owner";
    }
    else
    {
        if(numOwners == 1)
        {
            std::cout << "Owner: " << node->owners[0];
        }
        else
        {
            std::cout << "Owners: ";
            for(size_t i = 0; i < numOwners - 1; i++)
            {
                std::cout << node->owners[i] << ", ";
            }
            std::cout << node->owners[numOwners - 1];
        }
    }

    std::cout << ((node->is_leaf) ? " LEAF" : " NOT LEAF");

    std::cout << std::endl;

    for(const Node *child : node->children)
    {
        this->printHelper(child, tabs + 1);
    }
}

template<typename PointT, int max_ranks_per_leaf>
void HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::buildTreeHelper(
    Node *currentNode,
    const typename HilbertRectangularConvertor3D<PointT>::RecursionArguments &current_args,
    size_t currentDepth,
    hilbert_index_t &current_d,
    const std::shared_ptr<HilbertRectangularConvertor3D<PointT>> &rectConvertor,
    const std::vector<hilbert_index_t> &responsibilityRange)
{
    using DirectionVector3D = typename HilbertRectangularConvertor3D<PointT>::DirectionVector3D;
    using RecursionArguments = typename HilbertRectangularConvertor3D<PointT>::RecursionArguments;
    using direction_t = typename HilbertRectangularConvertor3D<PointT>::direction_t;

    if(currentNode == nullptr)
    {
        return;
    }

    this->depth = std::max(this->depth, currentDepth);

    const DirectionVector3D &startPoint = current_args.startPoint;
    const DirectionVector3D &a = current_args.a;
    const DirectionVector3D &b = current_args.b;
    const DirectionVector3D &c = current_args.c;
    direction_t width = std::abs(a.x + a.y + a.z);
    direction_t height = std::abs(b.x + b.y + b.z);
    direction_t depth = std::abs(c.x + c.y + c.z);

    size_t num_points = width * height * depth;

    currentNode->num_points = num_points;

    currentNode->d_start = current_d;
    currentNode->d_end = current_d + num_points;

    const std::pair<DirectionVector3D, DirectionVector3D> &boundingBox = rectConvertor->getBoundingBox(current_args);
    const DirectionVector3D &ll = boundingBox.first;
    const DirectionVector3D &ur = boundingBox.second;
    currentNode->boundingBox = BoundingBox<PointT>(rectConvertor->WidthHeightDepthToXYZ(ll.x, ll.y, ll.z),
                                                   rectConvertor->WidthHeightDepthToXYZ(ur.x, ur.y, ur.z));

    std::pair<int, int> ranksMatching = {0, this->size - 1};

    if(current_d >= responsibilityRange.back())
    {
        ranksMatching = {this->size - 1, this->size - 1};
    }
    else
    {
        for(int index = 0; index < this->size; index++)
        {
            if(currentNode->d_start <= responsibilityRange[index])
            {
                ranksMatching.first = index;
                break;
            }
        }

        for(int index = ranksMatching.first; index < this->size; index++)
        {
            if((currentNode->d_end - 1) <= responsibilityRange[index])
            {
                ranksMatching.second = index;
                break;
            }
        }
    }

    PointT newLL = std::numeric_limits<coord_type>::max() * PointT(1, 1, 1);
    PointT newUR = std::numeric_limits<coord_type>::lowest() * PointT(1, 1, 1);

    if((ranksMatching.second - ranksMatching.first) < max_ranks_per_leaf)
    {
        currentNode->is_leaf = true;
        size_t numOwners = ranksMatching.second - ranksMatching.first + 1;
        currentNode->owners.resize(numOwners);
        for(size_t i = 0; i < numOwners; i++)
        {
            currentNode->owners[i] = ranksMatching.first + i;
        }
        current_d += num_points;
    }
    else
    {
        currentNode->is_leaf = false;
        currentNode->owners.resize(0);

        direction_t dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
        direction_t dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
        direction_t dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

        bool baseCase = false;
        DirectionVector3D baseCaseUnitDirection;
        direction_t baseCaseLength;

        if(height == 1 and depth == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dax, day, daz};
            baseCaseLength = width;
        }

        if(width == 1 and depth == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dbx, dby, dbz};
            baseCaseLength = height;
        }

        if(width == 1 and height == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dcx, dcy, dcz};
            baseCaseLength = depth;
        }

        if(baseCase)
        {
            direction_t x = startPoint.x, y = startPoint.y, z = startPoint.z;
            for(int i = 0; i < baseCaseLength; i++)
            {
                currentNode->children.push_back(new Node(currentNode));
                this->buildTreeHelper(currentNode->children.back(),
                                      {{x, y, z}, {dax, day, daz}, {dbx, dby, dbz}, {dcx, dcy, dcz}},
                                      currentDepth + 1, current_d, rectConvertor, responsibilityRange);
                x += baseCaseUnitDirection.x;
                y += baseCaseUnitDirection.y;
                z += baseCaseUnitDirection.z;
            }
        }
        else
        {
            rectConvertor->setRecursionArguments(current_args, currentDepth);
            for(const RecursionArguments &nextArgs : rectConvertor->argumentsBuffer[currentDepth + 1])
            {
                currentNode->children.push_back(new Node(currentNode));
                this->buildTreeHelper(currentNode->children.back(), nextArgs, currentDepth + 1, current_d, rectConvertor, responsibilityRange);
            }

            for(const Node *child : currentNode->children)
            {
                const PointT &childLL = child->boundingBox.getLL();
                const PointT &childUR = child->boundingBox.getUR();

                for(int j = 0; j < DIM; j++)
                {
                    newLL[j] = std::min<coord_type>(newLL[j], childLL[j]);
                    newUR[j] = std::max<coord_type>(newUR[j], childUR[j]);
                }
            }
            newLL -= PointT(EPSILON, EPSILON, EPSILON);
            newUR += PointT(EPSILON, EPSILON, EPSILON);
            currentNode->boundingBox.setBounds(newLL, newUR);
        }
    }
}

template<typename PointT, int max_ranks_per_leaf>
void HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::buildTree(
    const std::shared_ptr<HilbertRectangularConvertor3D<PointT>> &rectConvertor,
    const std::vector<hilbert_index_t> &responsibilityRange)
{
    hilbert_index_t d = 0;

    if(responsibilityRange.empty())
    {
        throw DomainDecompError("Responsibility Range (Hilbert partition boundaries) given to tree is empty");
    }

    this->root = new Node(nullptr);
    this->buildTreeHelper(this->root,
                          {{0, 0, 0}, {rectConvertor->div.x, 0, 0}, {0, rectConvertor->div.y, 0}, {0, 0, rectConvertor->div.z}},
                          0, d, rectConvertor, responsibilityRange);

    if(d != rectConvertor->total_points_num)
    {
        DomainDecompError eo("HilbertRectangularTree3D::buildTree: d != convertor->total_points_num");
        eo.addEntry("d", d);
        eo.addEntry("total points num", rectConvertor->total_points_num);
        throw eo;
    }
}

template<typename PointT, int max_ranks_per_leaf>
template<typename U>
typename HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::RanksSet
HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::getIntersectingRanks(const Sphere<U> &sphere) const
{
    RanksSet result;
    if(this->root == nullptr || !SphereBoxIntersection(this->root->boundingBox, sphere))
    {
        return result;
    }
    this->nodes_stack.push_back(this->root);

    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node->is_leaf)
        {
            size_t numOwners = node->owners.size();
            for(size_t i = 0; i < numOwners; i++)
            {
                result.insert(node->owners[i]);
            }
        }
        else
        {
            for(const Node *child : node->children)
            {
                if(child != nullptr && SphereBoxIntersection(child->boundingBox, sphere))
                {
                    this->nodes_stack.push_back(child);
                }
            }
        }
    }

    if(result.empty())
    {
        DomainDecompError eo("In HilbertRectangularTree3D::getIntersectingRanks: result is empty, (should at least contain the rank itself)");
        eo.addEntry("Rank", rank);
        eo.addEntry("Sphere", sphere);
        eo.addEntry("Root Bounding Box", this->root->boundingBox);
        eo.addEntry("d of sphere center", this->convertor->xyz2d(sphere.center));
        this->print();
        throw eo;
    }

    return result;
}

template<typename PointT, int max_ranks_per_leaf>
template<typename U>
std::vector<std::pair<typename HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::coord_type,
                      typename HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::coord_type>>
HilbertRectangularTree3D<PointT, max_ranks_per_leaf>::getClosestFurthestPointsByRanks(const U &point) const
{
    const coord_type &maxVal = std::numeric_limits<coord_type>::max();
    const coord_type &minVal = std::numeric_limits<coord_type>::lowest();

    std::vector<std::pair<coord_type, coord_type>> distances(this->size, {maxVal, minVal});

    if(this->root != nullptr)
    {
        this->nodes_stack.push_back(this->root);
    }

    PointT closestPoint, furthestPoint;
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(!node->is_leaf)
        {
            for(const Node *child : node->children)
            {
                if(child != nullptr)
                {
                    this->nodes_stack.push_back(child);
                }
            }
            continue;
        }

        closestPoint = node->boundingBox.closestPoint(point);
        furthestPoint = node->boundingBox.furthestPoint(point);
        coord_type closestDist = 0, furthestDist = 0;
        for(int i = 0; i < DIM; i++)
        {
            closestDist += (closestPoint[i] - point[i]) * (closestPoint[i] - point[i]);
            furthestDist += (furthestPoint[i] - point[i]) * (furthestPoint[i] - point[i]);
        }

        size_t numOwners = node->owners.size();
        for(size_t i = 0; i < numOwners; i++)
        {
            int owner = node->owners[i];
            if(distances[owner].first > closestDist)
            {
                distances[owner].first = closestDist;
            }
            if(distances[owner].second < furthestDist)
            {
                distances[owner].second = furthestDist;
            }
        }
    }
    return distances;
}

template<typename PointT, int max_leaf_ranks>
std::vector<BoundingBox<PointT>> HilbertRectangularTree3D<PointT, max_leaf_ranks>::getRankBoundingBoxes(int _rank) const
{
    std::vector<BoundingBox<PointT>> result;

    this->nodes_stack.push_back(this->root);
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();
        if(node == nullptr)
        {
            continue;
        }
        if(node->is_leaf)
        {
            if(std::find(node->owners.begin(), node->owners.end(), _rank) != node->owners.end())
            {
                result.push_back(node->boundingBox);
            }
        }
        else
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child);
            }
        }
    }
    return result;
}

template<typename PointT, int max_leaf_ranks>
std::vector<std::vector<BoundingBox<PointT>>> HilbertRectangularTree3D<PointT, max_leaf_ranks>::getBoundingBoxesOfRanks(void) const
{
    std::vector<std::vector<BoundingBox<PointT>>> result(this->size);

    this->nodes_stack.push_back(this->root);
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();
        if(node == nullptr)
        {
            continue;
        }
        if(node->is_leaf)
        {
            for(const int &_rank : node->owners)
            {
                result[_rank].push_back(node->boundingBox);
            }
        }
        else
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child);
            }
        }
    }
    return result;
}

template<typename PointT, int max_leaf_ranks>
std::vector<const typename HilbertRectangularTree3D<PointT, max_leaf_ranks>::Node *>
HilbertRectangularTree3D<PointT, max_leaf_ranks>::getValuesIf(
    const std::function<bool(const Node *)> ifOpenFunction,
    const std::function<bool(const Node *)> &ifAddValueFunction) const
{
    std::vector<const Node *> nodes = {this->root};
    std::vector<const Node *> result;

    while(not nodes.empty())
    {
        const Node *node = nodes.back();
        nodes.pop_back();
        if(node == nullptr)
        {
            continue;
        }
        if(node->is_leaf)
        {
            if(ifAddValueFunction(node))
            {
                result.push_back(node);
            }
            continue;
        }
        if(ifOpenFunction(node))
        {
            for(const Node *child : node->children)
            {
                nodes.push_back(child);
            }
        }
    }
    return result;
}

template<typename PointT, int max_leaf_ranks>
std::vector<int> HilbertRectangularTree3D<PointT, max_leaf_ranks>::getOwners(const PointT &point) const
{
    std::vector<int> result;
    this->nodes_stack.push_back(this->root);

    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(node->is_leaf)
        {
            size_t numOwners = node->owners.size();
            for(size_t i = 0; i < numOwners; i++)
            {
                result.push_back(node->owners[i]);
            }
            continue;
        }
        else
        {
            if(node->boundingBox.isInside(point))
            {
                for(const Node *child : node->children)
                {
                    this->nodes_stack.push_back(child);
                }
            }
        }
    }
    return result;
}

#endif // RICH_MPI

#endif // HILBERT_RECTANGULAR_TREE_3D_HPP
