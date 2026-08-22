#ifndef MESH_DECOMPOSER_ONE_DIMENSIONAL_LOAD_BALANCER_HPP
#define MESH_DECOMPOSER_ONE_DIMENSIONAL_LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../balance/weightedBalance3.hpp"
#include "../error.hpp"
#include "LoadBalancer.hpp"

enum class Axis
{
    X,
    Y,
    Z
};

template<typename PointT>
class OneDimensionalLoadBalancer : public LoadBalancer<PointT>
{
public:
    static constexpr const char *type_name = "1d";

    OneDimensionalLoadBalancer(const PointT &ll, const PointT &ur, Axis axis, const MPI_Comm &comm = MPI_COMM_WORLD)
        : LoadBalancer<PointT>(comm), ll_(ll), ur_(ur), axis_(axis)
    {}

    OneDimensionalLoadBalancer(const PointT &ll, const PointT &ur, Axis axis, const std::vector<double> &bins, const MPI_Comm &comm = MPI_COMM_WORLD)
        : LoadBalancer<PointT>(comm), ll_(ll), ur_(ur), axis_(axis), bins_(bins)
    {}

    ~OneDimensionalLoadBalancer() override = default;

    std::shared_ptr<LoadBalancer<PointT>> clone() const override
    {
        return std::make_shared<OneDimensionalLoadBalancer<PointT>>(this->ll_, this->ur_, this->axis_, this->bins_, this->comm);
    }

    double Project(const PointT &point) const
    {
        switch(axis_)
        {
            case Axis::X: return point.x;
            case Axis::Y: return point.y;
            case Axis::Z: return point.z;
        }
        return point.x;
    }

    void rebalance(const std::vector<PointT> &points, const std::vector<double> &weights = std::vector<double>()) override
    {
        std::vector<double> projected;
        projected.reserve(points.size());
        for(const PointT &p : points)
        {
            projected.push_back(Project(p));
        }

        bins_ = getWeightedBorders3<double>(projected, weights, std::less<double>{}, this->comm);
    }

    int getOwner(const PointT &point) const override
    {
        if(bins_.empty())
        {
            return 0;
        }
        double coord = Project(point);
        size_t index = static_cast<size_t>(
            std::distance(bins_.cbegin(), std::upper_bound(bins_.cbegin(), bins_.cend(), coord)));
        return static_cast<int>(std::min<size_t>(index, static_cast<size_t>(this->size - 1)));
    }

    void changeBox(const std::pair<PointT, PointT> &newBox) override
    {
        if(bins_.empty())
        {
            ll_ = newBox.first;
            ur_ = newBox.second;
            return;
        }

        double oldLo = Project(ll_);
        double oldHi = Project(ur_);
        double oldLen = oldHi - oldLo;

        double newLo = Project(newBox.first);
        double newHi = Project(newBox.second);
        double newLen = newHi - newLo;

        if(oldLen > 0.0)
        {
            for(double &b : bins_)
            {
                b = newLo + (b - oldLo) / oldLen * newLen;
            }
        }

        ll_ = newBox.first;
        ur_ = newBox.second;
    }

    void printInfo() override
    {
        if(this->rank == 0)
        {
            const char *axisName = (axis_ == Axis::X) ? "X" : (axis_ == Axis::Y) ? "Y" : "Z";
            std::cout << "[OneDimensionalLoadBalancer] axis=" << axisName
                      << "  ranks=" << this->size
                      << "  bins=" << bins_.size() << std::endl;
            for(size_t i = 0; i < bins_.size(); ++i)
            {
                std::cout << "  bin[" << i << "] = " << bins_[i] << std::endl;
            }
        }
    }

    std::string getTypeName() const override { return type_name; }

    Axis GetAxis() const { return this->axis_; }
    const std::vector<double> &GetBins() const { return this->bins_; }
    const PointT &GetLowerLeft() const { return this->ll_; }
    const PointT &GetUpperRight() const { return this->ur_; }

    void SetBins(const std::vector<double> &bins)
    {
        this->bins_ = bins;
    }

private:
    PointT ll_, ur_;
    Axis axis_;
    std::vector<double> bins_;
};

#endif // RICH_MPI

#endif // MESH_DECOMPOSER_ONE_DIMENSIONAL_LOAD_BALANCER_HPP
