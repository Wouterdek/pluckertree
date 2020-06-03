#include <LBFGSB.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <MathUtil.h>
#include <fstream>
#include <iomanip>

#include <nlopt.hpp>
#include <Pluckertree.h>


using namespace LBFGSpp;

using Vector = Eigen::VectorXd;

class FixedMomentMinDist
{
private:
    using number_t = double;
    using Vector3_t = Eigen::Vector3d;
    using Vector2_t = Eigen::Vector2d;
    using Matrix3_t = Eigen::Matrix3d;

    Vector3_t q;
    Vector3_t h1;
    Vector3_t h2;
public:
    FixedMomentMinDist(Vector3_t q, Vector3_t dirLowerBound, Vector3_t dirUpperBound)
            : q(std::move(q)), h1(std::move(dirLowerBound)), h2(std::move(dirUpperBound)) {}

    double operator()(const Vector& x, Vector& grad)
    {
        // Careful, don't disturb the big pile of math!
        number_t phi_k = x[0];
        number_t theta_k = x[1];
        number_t r_k = x[2];

        // f(k)


        Vector3_t k = spherical2cart(phi_k, theta_k, r_k);
        Vector3_t kd = k.normalized();
        Vector3_t qp = q - kd*(q.dot(kd));
        Vector3_t qpd = qp.normalized();
        auto u = (qp - q).norm();

        auto sin_gamma = std::min(r_k/qp.norm(), (number_t)1.0f);
        auto cos_gamma = std::sqrt(1 - sin_gamma * sin_gamma);

        Vector3_t da = qpd * cos_gamma + kd.cross(qpd) * sin_gamma;
        Vector3_t d_alpha;
        auto da_dot_h1 = da.dot(h1);
        auto da_dot_h2 = da.dot(h2);
        if(da_dot_h1 >= 0 && da_dot_h2 >= 0)
        {
            d_alpha = da;
        }else if(da_dot_h1 <= da_dot_h2)
        {
            Vector h1_cross_k = h1.cross(k);
            d_alpha = std::copysign(1.0f, h2.dot(h1_cross_k)) * h1_cross_k.normalized();
        }else
        {
            Vector h2_cross_k = h2.cross(k);
            d_alpha = std::copysign(1.0f, h1.dot(h2_cross_k)) * h2_cross_k.normalized();
        }

        Vector3_t db = - qpd * cos_gamma + kd.cross(qpd) * sin_gamma;
        Vector3_t d_beta;
        auto db_dot_h1 = db.dot(h1);
        auto db_dot_h2 = db.dot(h2);
        if(db_dot_h1 >= 0 && db_dot_h2 >= 0)
        {
            d_beta = db;
        }else if(db_dot_h1 < db_dot_h2)
        {
            Vector h1_cross_k = h1.cross(k);
            d_beta = std::copysign(1.0f, h2.dot(h1_cross_k)) * h1_cross_k.normalized();
        }else
        {
            Vector h2_cross_k = h2.cross(k);
            d_beta = std::copysign(1.0f, h1.dot(h2_cross_k)) * h2_cross_k.normalized();
        }

        auto va = (qp.cross(d_alpha) - k).norm();
        auto vb = (qp.cross(d_beta) - k).norm();
        auto v = std::min(va, vb);

        auto f = std::sqrt(u*u + v*v);

        // partial
        /*auto sin_theta_k = std::sin(theta_k);
        auto pow2_sin_theta_k = sin_theta_k * sin_theta_k;
        auto sin_2theta_k = std::sin(2*theta_k);

        auto cos_theta_k = std::cos(theta_k);
        auto cos_2theta_k = std::cos(2*theta_k);
        auto pow2_cos_theta_k = cos_theta_k * cos_theta_k;

        auto sin_phi_k = std::sin(phi_k);
        auto pow2_sin_phi_k = sin_phi_k * sin_phi_k;
        auto sin_2phi_k = std::sin(2*phi_k);

        auto cos_phi_k = std::cos(phi_k);
        auto cos_2phi_k = std::cos(2*phi_k);
        auto pow2_cos_phi_k = cos_phi_k * cos_phi_k;

        ///
        auto partial_k_phi_k = r_k * Vector3_t(sin_theta_k * (-sin_phi_k), sin_theta_k * cos_phi_k, cos_theta_k);
        auto partial_k_theta_k = r_k * Vector3_t(cos_theta_k * cos_phi_k, cos_theta_k * sin_phi_k, (-sin_theta_k));
        auto partial_k_r_k = Vector3_t(sin_theta_k * cos_phi_k, sin_theta_k * sin_phi_k, cos_theta_k);

        Matrix3_t partial_k;
        partial_k.row(0) = partial_k_phi_k;
        partial_k.row(1) = partial_k_theta_k;
        partial_k.row(2) = partial_k_r_k;
        ///

        auto k_norm = k.norm();
        Matrix3_t partial_kd;
        for(int i = 0; i < 3; ++i)
        {
            partial_kd.row(i) = (partial_k.row(i)/k_norm) - k.transpose() * ((-1/(std::pow(k_norm, 3))) * (k.x() * partial_k.row(i).x() + k.y() * partial_k.row(i).y() + k.z() * partial_k.row(i).z()));
        }

        ///
        Matrix3_t partial_qp;
        for(int i = 0; i < 3; ++i)
        {
            partial_qp.row(i).x() = q.x() * 2 * kd.x() * partial_kd.row(i).x();
            partial_qp.row(i).x() += q.y() * (partial_kd.row(i).x() * kd.y() + kd.x() * partial_kd.row(i).y());
            partial_qp.row(i).x() += q.z() * (partial_kd.row(i).x() * kd.z() + kd.x() * partial_kd.row(i).z());

            partial_qp.row(i).y() = q.x() * (partial_kd.row(i).y() * kd.x() + kd.y() * partial_kd.row(i).x());
            partial_qp.row(i).y() += q.y() * 2 * kd.y() * partial_kd.row(i).y();
            partial_qp.row(i).y() += q.z() * (partial_kd.row(i).y() * kd.z() + kd.y() * partial_kd.row(i).z());

            partial_qp.row(i).z() = q.x() * (partial_kd.row(i).z() * kd.x() + kd.z() * partial_kd.row(i).x());
            partial_qp.row(i).z() += q.y() * (partial_kd.row(i).z() * kd.y() + kd.z() * partial_kd.row(i).y());
            partial_qp.row(i).z() += q.z() * 2 * kd.z() * partial_kd.row(i).z();

            partial_qp.row(i) *= -1;
        }
        ///

        //auto partial_pow2_u_phi_k = 2*(qp.x() - q.x()) * partial_qp_phi_k.x() + 2*(qp.y() - q.y()) * partial_qp_phi_k.y() + 2*(qp.z() - q.z()) * partial_qp_phi_k.z();
        auto partial_pow2_u = 2 * (partial_qp * (qp - q));



        auto da_xy_norm = Eigen::Vector2f(da.x(), da.y()).norm();

        auto qp_norm = qp.norm();
        //double partial_delta_phi_k = (1/qp_norm) * (qp.x() * partial_qp_phi_k.x() + qp.y() * partial_qp_phi_k.y() + qp.z() * partial_qp_phi_k.z());
        Vector3_t partial_delta;
        for(int i = 0; i < 3; i++)
        {
            partial_delta[i] = (1/qp_norm) * (qp.x() * partial_qp.row(i).x() + qp.y() * partial_qp.row(i).y() + qp.z() * partial_qp.row(i).z());
        }

        Matrix3_t partial_qpd;
        for(int i = 0; i < 3; i++)
        {
            partial_qpd.row(i) = (partial_qp.row(i)/qp_norm) + qp.transpose() * (-1/qp.squaredNorm()) * partial_delta[i];
        }



        Vector3_t partial_sin_gamma(0, 0, 0);
        //double partial_sin_gamma_phi_k = 1;
        if(r_k/qp_norm > 1)
        {
            for(int i = 0; i < 3; i++)
            {
                partial_sin_gamma[i] = -(r_k/qp.squaredNorm()) * partial_delta[i]; //TODO: is this wrong for i=2?
                //partial_sin_gamma_phi_k = -(r_k/qp.squaredNorm()) * partial_delta_phi_k;
            }
        }

        Vector3_t partial_cos_gamma;
        for(int i = 0; i < 3; i++)
        {
            //TODO
            partial_cos_gamma[i] = (1/std::sqrt(1-sin_gamma*sin_gamma)) * (-sin_gamma * partial_sin_gamma[i]);
        }

        Matrix3_t partial_da = Matrix3_t::Zero();
        float sign = 1.0f;
        if(va >= vb)
        {
            sign = -1.0f; //partial_db
        }

        for(int i = 0; i < 3; i++)
        {
            partial_da.row(i).x() += sign * partial_qpd.row(i).x() * cos_gamma + qpd.x() * partial_cos_gamma[i];
            partial_da.row(i).y() += sign * partial_qpd.row(i).y() * cos_gamma + qpd.x() * partial_cos_gamma[i];
            partial_da.row(i).z() += sign * partial_qpd.row(i).z() * cos_gamma + qpd.z() * partial_cos_gamma[i];

            partial_da.row(i).x() += partial_kd.row(i).y() * qpd.z() * sin_gamma + kd.y() * partial_qpd.row(i).z() * sin_gamma + kd.y() * qpd.z() * partial_sin_gamma[i];
            partial_da.row(i).y() += partial_kd.row(i).x() * qpd.z() * sin_gamma + kd.x() * partial_qpd.row(i).z() * sin_gamma + kd.x() * qpd.z() * partial_sin_gamma[i];
            partial_da.row(i).z() += partial_kd.row(i).x() * qpd.y() * sin_gamma + kd.x() * partial_qpd.row(i).y() * sin_gamma + kd.x() * qpd.y() * partial_sin_gamma[i];

            partial_da.row(i).x() += partial_kd.row(i).z() * qpd.y() * sin_gamma + kd.z() * partial_qpd.row(i).y() * sin_gamma + kd.z() * qpd.y() * partial_sin_gamma[i];
            partial_da.row(i).y() += partial_kd.row(i).z() * qpd.x() * sin_gamma + kd.z() * partial_qpd.row(i).x() * sin_gamma + kd.z() * qpd.x() * partial_sin_gamma[i];
            partial_da.row(i).z() += partial_kd.row(i).y() * qpd.x() * sin_gamma + kd.y() * partial_qpd.row(i).x() * sin_gamma + kd.y() * qpd.x() * partial_sin_gamma[i];
        }


        Matrix3_t partial_d_alpha; //also partial_d_beta, depending on va >= vb
        if(da_dot_h1 >= 0 && da_dot_h2 >= 0)
        {
            partial_d_alpha = partial_da;
        }else if(da_dot_h1 < da_dot_h2 || (da_dot_h1 == da_dot_h2 && va < vb))
        {
            for(int i = 0; i < 3; i++)
            {
                auto h1_cross_k_sqr_norm = h1.cross(k).squaredNorm();
                auto h1_cross_k_norm = std::sqrt(h1_cross_k_sqr_norm);

                auto partial_h1_cross_k_norm = (h1.y()*k.z() - h1.z()*k.y())*(h1.y()*partial_k.row(i).z() - h1.z()*partial_k.row(i).y());
                partial_h1_cross_k_norm += (h1.x()*k.z() - h1.z()*k.x())*(h1.x()*partial_k.row(i).z() - h1.z()*partial_k.row(i).x());
                partial_h1_cross_k_norm += (h1.x()*k.y() - h1.y()*k.x())*(h1.x()*partial_k.row(i).y() - h1.y()*partial_k.row(i).x());
                partial_h1_cross_k_norm *= (1.0/h1_cross_k_norm);

                auto fact = std::copysign(1.0, k.dot(h2.cross(h1))) / h1_cross_k_sqr_norm;

                partial_d_alpha.row(i).x() = h1_cross_k_norm * (h1.y() * partial_k.row(i).z() - h1.z() * partial_k.row(i).y());
                partial_d_alpha.row(i).x() -= partial_h1_cross_k_norm * (h1.y() * k.z() - h1.z() * k.y());

                partial_d_alpha.row(i).y() = h1_cross_k_norm * (h1.x() * partial_k.row(i).z() - h1.z() * partial_k.row(i).x());
                partial_d_alpha.row(i).y() -= partial_h1_cross_k_norm * (h1.x() * k.z() - h1.z() * k.x());

                partial_d_alpha.row(i).z() = h1_cross_k_norm * (h1.x() * partial_k.row(i).y() - h1.y() * partial_k.row(i).x());
                partial_d_alpha.row(i).z() -= partial_h1_cross_k_norm * (h1.x() * k.y() - h1.y() * k.x());

                partial_d_alpha.row(i) *= fact;
            }
        }else
        {
            for(int i = 0; i < 3; i++)
            {
                auto h2_cross_k_sqr_norm = h2.cross(k).squaredNorm();
                auto h2_cross_k_norm = std::sqrt(h2_cross_k_sqr_norm);

                auto partial_h2_cross_k_norm = (h1.y()*k.z() - h2.z()*k.y())*(h2.y()*partial_k.row(i).z() - h2.z()*partial_k.row(i).y());
                partial_h2_cross_k_norm += (h2.x()*k.z() - h2.z()*k.x())*(h2.x()*partial_k.row(i).z() - h2.z()*partial_k.row(i).x());
                partial_h2_cross_k_norm += (h2.x()*k.y() - h2.y()*k.x())*(h2.x()*partial_k.row(i).y() - h2.y()*partial_k.row(i).x());
                partial_h2_cross_k_norm *= (1.0/h2_cross_k_norm);

                auto fact = std::copysign(1.0, k.dot(h2.cross(h1))) / h2_cross_k_sqr_norm;

                partial_d_alpha.row(i).x() = h2_cross_k_norm * (h2.y() * partial_k.row(i).z() - h2.z() * partial_k.row(i).y());
                partial_d_alpha.row(i).x() -= partial_h2_cross_k_norm * (h2.y() * k.z() - h2.z() * k.y());

                partial_d_alpha.row(i).y() = h2_cross_k_norm * (h2.x() * partial_k.row(i).z() - h2.z() * partial_k.row(i).x());
                partial_d_alpha.row(i).y() -= partial_h2_cross_k_norm * (h2.x() * k.z() - h2.z() * k.x());

                partial_d_alpha.row(i).z() = h2_cross_k_norm * (h2.x() * partial_k.row(i).y() - h2.y() * partial_k.row(i).x());
                partial_d_alpha.row(i).z() -= partial_h2_cross_k_norm * (h2.x() * k.y() - h2.y() * k.x());

                partial_d_alpha.row(i) *= fact;
            }
        }

        ////

        auto c_alpha = qp.cross(da);
        Matrix3_t partial_c_alpha;
        for(int i = 0; i < 3; i++)
        {
            partial_c_alpha.row(i) = Vector3_t(
                    (partial_qp.row(i).y()*d_alpha.z() + qp.y() * partial_d_alpha.row(i).z()) - (partial_qp.row(i).z()*d_alpha.y() + qp.z()*partial_d_alpha.row(i).y()),
                    (partial_qp.row(i).x()*d_alpha.z() + qp.x() * partial_d_alpha.row(i).z()) - (partial_qp.row(i).z()*d_alpha.x() + qp.z()*partial_d_alpha.row(i).x()),
                    (partial_qp.row(i).x()*d_alpha.y() + qp.x() * partial_d_alpha.row(i).y()) - (partial_qp.row(i).y()*d_alpha.x() + qp.y()*partial_d_alpha.row(i).x())
            );
        }

        Vector3_t partial_v;
        for(int i = 0; i < 3; i++)
        {
            partial_v[i] = (1.0/va) * (
                    (c_alpha.x() - k.x()) * (partial_c_alpha.row(i).x() - partial_k.row(i).x()) +
                    (c_alpha.y() - k.y()) * (partial_c_alpha.row(i).y() - partial_k.row(i).y()) +
                    (c_alpha.z() - k.z()) * (partial_c_alpha.row(i).z() - partial_k.row(i).z())
            );
        }

        Vector3_t partial_pow2_v = 2*va*partial_v;

        grad = (1.0/(2.0*std::sqrt(u*u + v*v))) * (partial_pow2_u + partial_pow2_v);
        /*assert(!grad.hasNaN());
        assert(!std::isinf(grad[0]));
        assert(!std::isinf(grad[1]));
        assert(!std::isinf(grad[2]));*/

        return f;
    }
};

class FixedMomentMinHitDist
{
private:
    using number_t = double;
    using Vector3_t = Eigen::Vector3d;
    using Vector2_t = Eigen::Vector2d;
    using Matrix3_t = Eigen::Matrix3d;

    Vector3_t q;
    Vector3_t q_n;
    Vector3_t h1;
    Vector3_t h2;
public:
    FixedMomentMinHitDist(Vector3_t q, Vector3_t q_n, Vector3_t dirLowerBound, Vector3_t dirUpperBound)
            : q(std::move(q)), q_n(std::move(q_n)), h1(std::move(dirLowerBound)), h2(std::move(dirUpperBound)) {}

    double operator()(const Vector& x, Vector& grad)
    {
        // Careful, don't disturb the big pile of math!
        number_t phi_k = x[0];
        number_t theta_k = x[1];
        number_t r_k = x[2];

        // f(k)


        Vector3_t k = spherical2cart(phi_k, theta_k, r_k);
        Vector3_t kd = k.normalized();
        Vector3_t qp = q - kd*(q.dot(kd));
        Vector3_t qpd = qp.normalized();
        auto u = (qp - q).norm();

        auto sin_gamma = std::min(r_k/qp.norm(), (number_t)1.0f);
        auto cos_gamma = std::sqrt(1 - sin_gamma * sin_gamma);

        Vector3_t da = qpd * cos_gamma + kd.cross(qpd) * sin_gamma;
        Vector3_t d_alpha;
        auto da_dot_h1 = da.dot(h1);
        auto da_dot_h2 = da.dot(h2);
        if(da_dot_h1 >= 0 && da_dot_h2 >= 0)
        {
            d_alpha = da;
        }else if(da_dot_h1 <= da_dot_h2)
        {
            Vector h1_cross_k = h1.cross(k);
            d_alpha = std::copysign(1.0f, h2.dot(h1_cross_k)) * h1_cross_k.normalized();
        }else
        {
            Vector h2_cross_k = h2.cross(k);
            d_alpha = std::copysign(1.0f, h1.dot(h2_cross_k)) * h2_cross_k.normalized();
        }

        Vector3_t db = - qpd * cos_gamma + kd.cross(qpd) * sin_gamma;
        Vector3_t d_beta;
        auto db_dot_h1 = db.dot(h1);
        auto db_dot_h2 = db.dot(h2);
        if(db_dot_h1 >= 0 && db_dot_h2 >= 0)
        {
            d_beta = db;
        }else if(db_dot_h1 < db_dot_h2)
        {
            Vector h1_cross_k = h1.cross(k);
            d_beta = std::copysign(1.0f, h2.dot(h1_cross_k)) * h1_cross_k.normalized();
        }else
        {
            Vector h2_cross_k = h2.cross(k);
            d_beta = std::copysign(1.0f, h1.dot(h2_cross_k)) * h2_cross_k.normalized();
        }

        auto va = (qp.cross(d_alpha) - k).norm();
        auto vb = (qp.cross(d_beta) - k).norm();
        auto v = std::min(va, vb);

        auto f = std::sqrt(u*u + v*v);

        return f;
    }
};

namespace pluckertree
{
    int pluckertree::TreeNode::visited = 0;
    std::vector<float> pluckertree::TreeNode::results;

    double FindMinDist(
            const Eigen::Vector3f& point,
            const Eigen::Vector3f& dirLowerBound,
            const Eigen::Vector3f& dirUpperBound,
            const Eigen::Vector3f& momentLowerBound,
            const Eigen::Vector3f& momentUpperBound,
            Eigen::Vector3f& minimum
    )
    {

        nlopt::opt opt(nlopt::LN_COBYLA, 3);//LN_NELDERMEAD, LN_SBPLX

        std::vector<double> lb(momentLowerBound.data(), momentLowerBound.data() + momentLowerBound.rows() * momentLowerBound.cols());
        opt.set_lower_bounds(lb);

        std::vector<double> hb(momentUpperBound.data(), momentUpperBound.data() + momentUpperBound.rows() * momentUpperBound.cols());
        opt.set_upper_bounds(hb);

        FixedMomentMinDist fun(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());
        auto obj_func = [](const std::vector<double> &x, std::vector<double> &grad, void* f_data) -> double {
            FixedMomentMinDist* fun = reinterpret_cast<FixedMomentMinDist*>(f_data);
            Vector x_vect = Eigen::Vector3d {x[0], x[1], x[2]};
            Vector grad_vect = Eigen::Vector3d {0, 0, 0};
            auto result = (*fun)(x_vect, grad_vect);
            pluckertree::TreeNode::results.push_back(result);
            return result;
        };
        opt.set_min_objective(obj_func, &fun);

        opt.set_xtol_rel(1e-6);
        opt.set_stopval(1e-6);

        Vector vec = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();
        std::vector<double> x(vec.data(), vec.data() + vec.rows() * vec.cols());

        try
        {
            double minf;
            nlopt::result result = opt.optimize(x, minf);

            minimum = {x[0], x[1], x[2]};
            return minf;
        }
        catch(std::exception &e) {
            std::cout << "nlopt failed: " << e.what() << std::endl;
            throw;
        }

        /*// Set up parameters
        LBFGSBParam<double> param;
        param.epsilon = 1e-6;
        param.max_iterations = 100;

        // Create solver and function object
        LBFGSBSolver<double> solver(param);  // New solver class
        FixedMomentMinDist fun(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

        // Initial guess
        Vector x = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();

        // x will be overwritten to be the best point found
        double dist;
        int nbIter = solver.minimize(fun, x, dist, momentLowerBound.cast<double>(), momentUpperBound.cast<double>());
        minimum = x.cast<float>();

        std::cout << nbIter << " iterations" << std::endl;
        std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << dist << std::endl;

        return dist;*/
    }

    double FindMinHitDist(
            const Eigen::Vector3f& point,
            const Eigen::Vector3f& point_normal,
            const Eigen::Vector3f& dirLowerBound,
            const Eigen::Vector3f& dirUpperBound,
            const Eigen::Vector3f& momentLowerBound,
            const Eigen::Vector3f& momentUpperBound,
            Eigen::Vector3f& minimum
    )
    {

        nlopt::opt opt(nlopt::LN_COBYLA, 3);//LN_NELDERMEAD, LN_SBPLX

        std::vector<double> lb(momentLowerBound.data(), momentLowerBound.data() + momentLowerBound.rows() * momentLowerBound.cols());
        opt.set_lower_bounds(lb);

        std::vector<double> hb(momentUpperBound.data(), momentUpperBound.data() + momentUpperBound.rows() * momentUpperBound.cols());
        opt.set_upper_bounds(hb);

        FixedMomentMinHitDist fun(point.cast<double>(), point_normal.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());
        auto obj_func = [](const std::vector<double> &x, std::vector<double> &grad, void* f_data) -> double {
            FixedMomentMinHitDist* fun = reinterpret_cast<FixedMomentMinHitDist*>(f_data);
            Vector x_vect = Eigen::Vector3d {x[0], x[1], x[2]};
            Vector grad_vect = Eigen::Vector3d {0, 0, 0};
            auto result = (*fun)(x_vect, grad_vect);
            pluckertree::TreeNode::results.push_back(result);
            return result;
        };
        opt.set_min_objective(obj_func, &fun);

        opt.set_xtol_rel(1e-6);
        opt.set_stopval(1e-6);

        Vector vec = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();
        std::vector<double> x(vec.data(), vec.data() + vec.rows() * vec.cols());

        try
        {
            double minf;
            nlopt::result result = opt.optimize(x, minf);

            minimum = {x[0], x[1], x[2]};
            return minf;
        }
        catch(std::exception &e) {
            std::cout << "nlopt failed: " << e.what() << std::endl;
            throw;
        }
    }

    double FindMinDist(
            const Eigen::Vector3f& point,
            const Eigen::Vector3f& dirLowerBound,
            const Eigen::Vector3f& dirUpperBound,
            const Eigen::Vector3f& moment
    )
    {
        FixedMomentMinDist f(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

        Vector vect = moment.cast<double>();
        Vector grad;
        auto dist = f(vect, grad);

        return dist;
    }

///// POINTCLOUD EXPORT

struct GridPoint
{
    Eigen::Vector3f pos;
    Eigen::Vector3f grad;
    float dist;
};

std::vector<GridPoint> CalculateGrid(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& dirLowerBound,
        const Eigen::Vector3f& dirUpperBound,
        float x_start,
        float x_end,
        float y_start,
        float y_end,
        float z_start,
        float z_end,
        float resolution
)
{
    FixedMomentMinDist f(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

    auto x_steps = int((x_end - x_start) / resolution);
    auto y_steps = int((y_end - y_start) / resolution);
    auto z_steps = int((z_end - z_start) / resolution);

    std::vector<GridPoint> points(x_steps * y_steps * z_steps);

    for(int x_i = 0; x_i < x_steps; x_i++)
    {
        auto x = x_start + resolution*x_i;
        for(int y_i = 0; y_i < y_steps; y_i++)
        {
            auto y = y_start + resolution*y_i;
            for(int z_i = 0; z_i < z_steps; z_i++)
            {
                auto z = z_start + resolution*z_i;

                Vector vect = Eigen::Vector3d(x, y, z);
                Vector grad;
                auto dist = f(vect, grad);

                auto& curPoint = points[(x_i * y_steps * z_steps) + (y_i * z_steps) + z_i];
                curPoint.pos = Eigen::Vector3f(x, y, z);
                curPoint.dist = dist;
                curPoint.grad = grad.cast<float>();
            }
        }
    }

    return points;
}


void show_me_the_grid()
{
    /*const Eigen::Vector3f point(1, 0, 0);
    Eigen::Vector3f dirLowerBound(-M_PI, 0);
    Eigen::Vector3f dirUpperBound(M_PI, M_PI);
    float x_start = -1;
    float x_end = 1;
    float y_start = -1;
    float y_end = 1;
    float z_start = -1;
    float z_end = 1;
    float resolution = 0.1;
    auto data = CalculateGrid(point, dirLowerBound, dirUpperBound, x_start, x_end, y_start, y_end, z_start, z_end, resolution);

    std::fstream myfile;
    myfile = std::fstream("/home/wouter/Desktop/pluckerdata", std::ios::out | std::ios::binary);

    unsigned int size = data.size();
    myfile.write(reinterpret_cast<char*>(&size), sizeof(unsigned int));
    for(const auto& entry : data)
    {
        myfile.write(reinterpret_cast<const char*>(&entry.pos.x()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.pos.y()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.pos.z()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.grad.x()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.grad.y()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.grad.z()), sizeof(float));
        myfile.write(reinterpret_cast<const char*>(&entry.dist), sizeof(float));
    }
    myfile.close();*/
}

/////

}