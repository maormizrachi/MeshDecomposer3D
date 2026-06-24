#ifndef HILBERT_RECTANGULAR_CONVERTOR_3D_HPP
#define HILBERT_RECTANGULAR_CONVERTOR_3D_HPP

#include <array>
#include <cmath>
#include <utility>
#include <spatial_ds/utils/geometry.hpp>
#include "../HilbertConvertor3D.hpp"
#include "../../error.hpp"

#ifndef SIGN
#define SIGN(x) ((x > 0) - (x < 0))
#endif

#define MAX_HILBERT_DEPTH 54

template<typename PointT, int max_leaf_ranks>
class HilbertRectangularTree3D;

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
 */
template<typename PointT>
class HilbertRectangularConvertor3D : public HilbertConvertor3D<PointT>
{
    template<typename PT, int max_leaf_ranks>
    friend class HilbertRectangularTree3D;

public:
    using coord_t = typename PointT::coord_type;

private:
    using direction_t = long int;

    struct DirectionVector3D
    {
        direction_t x, y, z;

#ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const DirectionVector3D &args)
        {
            return os << "(" << args.x << ", " << args.y << ", " << args.z << ")";
        }
#endif // DEBUG_MODE
    };

    struct RecursionArguments
    {
        DirectionVector3D startPoint;
        DirectionVector3D a;
        DirectionVector3D b;
        DirectionVector3D c;

#ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const RecursionArguments &args)
        {
            return os << "startPoint = " << args.startPoint << ", a = " << args.a << ", b = " << args.b << ", c = " << args.c;
        }
#endif // DEBUG_MODE
    };

    PointT step;
    BoundingBox<PointT> spaceBoundingBox;
    DirectionVector3D div;
    hilbert_index_t total_points_num;

    mutable std::array<std::vector<RecursionArguments>, MAX_HILBERT_DEPTH> argumentsBuffer;

public:
    explicit HilbertRectangularConvertor3D(const PointT &ll, const PointT &ur, size_t order);

    ~HilbertRectangularConvertor3D() override = default;

    static constexpr const char *type_name = "rectangular";

    std::string getTypeName() const override { return type_name; }

    inline hilbert_index_t getHilbertSize() const { return this->total_points_num; }

    void changeOrder(size_t order) override;

    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const override;

    PointT d2xyz(hilbert_index_t d) const override;

    inline std::shared_ptr<HilbertConvertor3D<PointT>> clone(void) const override
    {
        return std::make_shared<HilbertRectangularConvertor3D>(this->ll, this->ur, this->order);
    }

private:
    void setRecursionArguments(const RecursionArguments &args, size_t currentDepth) const;

    bool d2xyz_helper(const RecursionArguments &args, size_t currentDepth, hilbert_index_t requested_d, hilbert_index_t &current_d, PointT &result) const;

    bool xyz2d_helper_base(const DirectionVector3D &startPoint, size_t steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;

    bool xyz2d_helper(const RecursionArguments &args, size_t currentDepth, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;

    std::pair<DirectionVector3D, DirectionVector3D> getBoundingBox(const RecursionArguments &args) const;

    PointT WidthHeightDepthToXYZ(direction_t width, direction_t height, direction_t depth) const;
};

template<typename PointT>
HilbertRectangularConvertor3D<PointT>::HilbertRectangularConvertor3D(const PointT &ll, const PointT &ur, size_t order)
    : HilbertConvertor3D<PointT>(ll, ur, order)
{
    this->spaceBoundingBox = BoundingBox<PointT>(ll, ur);
    this->changeOrder(order);
}

template<typename PointT>
void HilbertRectangularConvertor3D<PointT>::changeOrder(size_t order)
{
    this->order = order = std::min<size_t>(MAX_HILBERT_ORDER, order);
    coord_t realWidth = this->ur.x - this->ll.x;
    coord_t realHeight = this->ur.y - this->ll.y;
    coord_t realDepth = this->ur.z - this->ll.z;

    this->div.x = std::ceil(std::pow((realWidth * realWidth) / (realHeight * realDepth), 0.333333333) * std::pow(2, order));
    this->div.y = std::ceil(std::pow((realHeight * realHeight) / (realWidth * realDepth), 0.333333333) * std::pow(2, order));
    this->div.z = std::ceil(std::pow(8, order) / (this->div.x * this->div.y));

    this->total_points_num = this->div.x * this->div.y * this->div.z;
    this->step = PointT(realWidth / div.x, realHeight / div.y, realDepth / div.z);
}

template<typename PointT>
void HilbertRectangularConvertor3D<PointT>::setRecursionArguments(const RecursionArguments &args, size_t currentDepth) const
{
    const DirectionVector3D &startPoint = args.startPoint;
    const DirectionVector3D &a = args.a;
    const DirectionVector3D &b = args.b;
    const DirectionVector3D &c = args.c;

    direction_t width = std::abs(a.x + a.y + a.z);
    direction_t height = std::abs(b.x + b.y + b.z);
    direction_t depth = std::abs(c.x + c.y + c.z);

    direction_t dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    direction_t dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    DirectionVector3D a2 = {a.x >> 1, a.y >> 1, a.z >> 1};
    DirectionVector3D b2 = {b.x >> 1, b.y >> 1, b.z >> 1};
    DirectionVector3D c2 = {c.x >> 1, c.y >> 1, c.z >> 1};

    direction_t width2 = std::abs(a2.x + a2.y + a2.z);
    direction_t height2 = std::abs(b2.x + b2.y + b2.z);
    direction_t depth2 = std::abs(c2.x + c2.y + c2.z);

    if((width2 % 2) and (width > 2))
    {
        a2.x = a2.x + dax;
        a2.y = a2.y + day;
        a2.z = a2.z + daz;
    }

    if((height2 % 2) and (height > 2))
    {
        b2.x = b2.x + dbx;
        b2.y = b2.y + dby;
        b2.z = b2.z + dbz;
    }

    if((depth2 % 2) and (depth > 2))
    {
        c2.x = c2.x + dcx;
        c2.y = c2.y + dcy;
        c2.z = c2.z + dcz;
    }

    const direction_t &x = startPoint.x;
    const direction_t &y = startPoint.y;
    const direction_t &z = startPoint.z;

    if((currentDepth + 1) >= MAX_HILBERT_DEPTH)
    {
        throw DomainDecompError("Max hilbert tree depth exceeded");
    }
    std::vector<RecursionArguments> &nextArgs = this->argumentsBuffer[currentDepth + 1];
    nextArgs.clear();

    if((2 * width > 3 * height) and (2 * width > 3 * depth))
    {
        nextArgs.push_back({startPoint, a2, b, c});
        nextArgs.push_back({{x + a2.x, y + a2.y, z + a2.z}, {a.x - a2.x, a.y - a2.y, a.z - a2.z}, b, c});
    }
    else if(3 * height > 4 * depth)
    {
        nextArgs.push_back({startPoint, b2, c, a2});
        nextArgs.push_back({{x + b2.x, y + b2.y, z + b2.z}, a, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, c});
        nextArgs.push_back({{x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}});
    }
    else if(3 * depth > 4 * height)
    {
        nextArgs.push_back({startPoint, c2, a2, b});
        nextArgs.push_back({{x + c2.x, y + c2.y, z + c2.z}, a, b, {c.x - c2.x, c.y - c2.y, c.z - c2.z}});
        nextArgs.push_back({{x + (a.x - dax) + (c2.x - dcx), y + (a.y - day) + (c2.y - dcy), z + (a.z - daz) + (c2.z - dcz)}, {-c2.x, -c2.y, -c2.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, b});
    }
    else
    {
        nextArgs.push_back({startPoint, b2, c2, a2});
        nextArgs.push_back({{x + b2.x, y + b2.y, z + b2.z}, c, a2, {b.x - b2.x, b.y - b2.y, b.z - b2.z}});
        nextArgs.push_back({{x + (b2.x - dbx) + (c.x - dcx), y + (b2.y - dby) + (c.y - dcy), z + (b2.z - dbz) + (c.z - dcz)}, a, {-b2.x, -b2.y, -b2.z}, {-(c.x - c2.x), -(c.y - c2.y), -(c.z - c2.z)}});
        nextArgs.push_back({{x + (a.x - dax) + b2.x + (c.x - dcx), y + (a.y - day) + b2.y + (c.y - dcy), z + (a.z - daz) + b2.z + (c.z - dcz)}, {-c.x, -c.y, -c.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, {b.x - b2.x, b.y - b2.y, b.z - b2.z}});
        nextArgs.push_back({{x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c2, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}});
    }
}

template<typename PointT>
PointT HilbertRectangularConvertor3D<PointT>::WidthHeightDepthToXYZ(direction_t width, direction_t height, direction_t depth) const
{
    coord_t x = this->ll[0] + width * this->step[0];
    coord_t y = this->ll[1] + height * this->step[1];
    coord_t z = this->ll[2] + depth * this->step[2];
    return PointT(x, y, z);
}

template<typename PointT>
bool HilbertRectangularConvertor3D<PointT>::d2xyz_helper(const RecursionArguments &args, size_t currentDepth, hilbert_index_t requested_d, hilbert_index_t &current_d, PointT &result) const
{
    const DirectionVector3D &startPoint = args.startPoint;
    const DirectionVector3D &a = args.a;
    const DirectionVector3D &b = args.b;
    const DirectionVector3D &c = args.c;

    direction_t width = std::abs(a.x + a.y + a.z);
    direction_t height = std::abs(b.x + b.y + b.z);
    direction_t depth = std::abs(c.x + c.y + c.z);

    size_t num_points = width * height * depth;

    direction_t dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    direction_t dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    if(requested_d >= current_d + num_points)
    {
        current_d += num_points;
        return false;
    }

    if(requested_d < current_d)
    {
        throw DomainDecompError("in HilbertRectangularConvertor3D::d2xyz_helper, should not reach here (algorithm failed)");
    }
    hilbert_index_t diff = requested_d - current_d;

    if(height == 1 and depth == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dax, startPoint.y + diff * day, startPoint.z + diff * daz);
        return true;
    }

    if(width == 1 and depth == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dbx, startPoint.y + diff * dby, startPoint.z + diff * dbz);
        return true;
    }

    if(width == 1 and height == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dcx, startPoint.y + diff * dcy, startPoint.z + diff * dcz);
        return true;
    }

    this->setRecursionArguments(args, currentDepth);
    for(const RecursionArguments &nextArgs : this->argumentsBuffer[currentDepth + 1])
    {
        if(this->d2xyz_helper(nextArgs, currentDepth + 1, requested_d, current_d, result))
        {
            return true;
        }
    }

    return false;
}

template<typename PointT>
bool HilbertRectangularConvertor3D<PointT>::xyz2d_helper_base(const DirectionVector3D &startPoint, size_t steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const
{
    direction_t x = startPoint.x, y = startPoint.y, z = startPoint.z;
    for(size_t i = 0; i < steps; i++)
    {
        if((requested_point.x == x) and (requested_point.y == y) and (requested_point.z == z))
        {
            return true;
        }
        x += direction.x;
        y += direction.y;
        z += direction.z;
        current_d++;
    }
    return false;
}

template<typename PointT>
std::pair<typename HilbertRectangularConvertor3D<PointT>::DirectionVector3D, typename HilbertRectangularConvertor3D<PointT>::DirectionVector3D>
HilbertRectangularConvertor3D<PointT>::getBoundingBox(const RecursionArguments &args) const
{
    const DirectionVector3D &startPoint = args.startPoint;
    const DirectionVector3D &a = args.a;
    const DirectionVector3D &b = args.b;
    const DirectionVector3D &c = args.c;

    direction_t x_advancing = a.x + b.x + c.x;
    direction_t y_advancing = a.y + b.y + c.y;
    direction_t z_advancing = a.z + b.z + c.z;

    DirectionVector3D boundary = {startPoint.x + x_advancing + ((x_advancing >= 0) ? 1 : 0),
                                  startPoint.y + y_advancing + ((y_advancing >= 0) ? 1 : 0),
                                  startPoint.z + z_advancing + ((z_advancing >= 0) ? 1 : 0)};
    return {{std::min(startPoint.x, boundary.x) - 1, std::min(startPoint.y, boundary.y) - 1, std::min(startPoint.z, boundary.z) - 1},
            {std::max(startPoint.x, boundary.x) + 1, std::max(startPoint.y, boundary.y) + 1, std::max(startPoint.z, boundary.z) + 1}};
}

template<typename PointT>
bool HilbertRectangularConvertor3D<PointT>::xyz2d_helper(const RecursionArguments &args, size_t currentDepth, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const
{
    const DirectionVector3D &startPoint = args.startPoint;
    const DirectionVector3D &a = args.a;
    const DirectionVector3D &b = args.b;
    const DirectionVector3D &c = args.c;

    direction_t width = std::abs(a.x + a.y + a.z);
    direction_t height = std::abs(b.x + b.y + b.z);
    direction_t depth = std::abs(c.x + c.y + c.z);

    size_t num_points = width * height * depth;

    std::pair<DirectionVector3D, DirectionVector3D> bounding_box = this->getBoundingBox(args);
    if((requested_point.x < bounding_box.first.x) or (requested_point.x > bounding_box.second.x) or
       (requested_point.y < bounding_box.first.y) or (requested_point.y > bounding_box.second.y) or
       (requested_point.z < bounding_box.first.z) or (requested_point.z > bounding_box.second.z))
    {
        current_d += num_points;
        return false;
    }

    direction_t dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    direction_t dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    if(height == 1 and depth == 1)
    {
        return this->xyz2d_helper_base(startPoint, width, {dax, day, daz}, requested_point, current_d);
    }

    if(width == 1 and depth == 1)
    {
        return this->xyz2d_helper_base(startPoint, height, {dbx, dby, dbz}, requested_point, current_d);
    }

    if(width == 1 and height == 1)
    {
        return this->xyz2d_helper_base(startPoint, depth, {dcx, dcy, dcz}, requested_point, current_d);
    }

    this->setRecursionArguments(args, currentDepth);
    for(const RecursionArguments &nextArgs : this->argumentsBuffer[currentDepth + 1])
    {
        if(this->xyz2d_helper(nextArgs, currentDepth + 1, requested_point, current_d))
        {
            return true;
        }
    }
    return false;
}

template<typename PointT>
PointT HilbertRectangularConvertor3D<PointT>::d2xyz(hilbert_index_t d) const
{
    PointT result;
    hilbert_index_t current_d = 0;
    this->d2xyz_helper({{0, 0, 0}, {this->div.x, 0, 0}, {0, this->div.y, 0}, {0, 0, this->div.z}}, 0, d, current_d, result);
    return result;
}

template<typename PointT>
hilbert_index_t HilbertRectangularConvertor3D<PointT>::xyz2d(coord_t x, coord_t y, coord_t z) const
{
    direction_t width = std::floor((x - this->ll.x) / this->step.x);
    direction_t height = std::floor((y - this->ll.y) / this->step.y);
    direction_t depth = std::floor((z - this->ll.z) / this->step.z);

    if((width < 0) or (height < 0) or (depth < 0) or (width > this->div.x) or (height > this->div.y) or (depth > this->div.z))
    {
        DomainDecompError eo("Should not reach here, overflow (in 3D xyz->d)");
        eo.addEntry("Width", width);
        eo.addEntry("this->div.x", this->div.x);
        eo.addEntry("Height", height);
        eo.addEntry("this->div.y", this->div.y);
        eo.addEntry("Depth", depth);
        eo.addEntry("this->div.z", this->div.z);
        eo.addEntry("x", x);
        eo.addEntry("y", y);
        eo.addEntry("z", z);
        eo.addEntry("step", this->step);
        eo.addEntry("ll", this->ll);
        throw eo;
    }

    hilbert_index_t result = 0;
    if(not this->xyz2d_helper({{0, 0, 0}, {this->div.x, 0, 0}, {0, this->div.y, 0}, {0, 0, this->div.z}}, 0, {width, height, depth}, result))
    {
        DomainDecompError eo("Should not reach here (in 3D xyz->d) (maybe the point is outside the box?)");
        eo.addEntry("x", x);
        eo.addEntry("y", y);
        eo.addEntry("z", z);
        throw eo;
    }
    return result;
}

#endif // HILBERT_RECTANGULAR_CONVERTOR_3D_HPP
