#include <LBFGSB.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <MathUtil.h>
#include <fstream>
#include <iomanip>

#include <nlopt.hpp>
#include <Pluckertree.h>
#include <PluckertreeSegments.h>

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
        Vector3_t kd = k / r_k;
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

        //Find intersection line of directionvector plane with query plane
        Vector3_t kd_cross_qn = kd.cross(q_n);
        auto kd_cross_qn_norm = kd_cross_qn.norm();
        if(kd_cross_qn_norm < 1e-6) // All lines are parallel to the query plane, assume all miss.
        {
            return std::numeric_limits<double>::infinity();
        }
        Vector3_t isect_d = kd_cross_qn / kd_cross_qn_norm;
        Vector3_t isect_k = kd * (q_n.dot(q)/kd_cross_qn_norm);
        Vector3_t isect_p = isect_d.cross(isect_k);
        Vector3_t qpl = isect_p + isect_d * isect_d.dot(q - isect_p);
        Vector3_t qpld = qpl.normalized();

        auto sin_gamma = std::min(r_k/qpl.norm(), (number_t)1.0f);
        auto cos_gamma = std::sqrt(1 - sin_gamma * sin_gamma);

        Vector3_t da = qpld * cos_gamma + kd.cross(qpld) * sin_gamma;
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

        Vector3_t db = - qpld * cos_gamma + kd.cross(qpld) * sin_gamma;
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

        //Calculate intersection point of isect line with (d;k)
        number_t va = std::numeric_limits<double>::infinity();
        if(1 - std::abs(d_alpha.dot(isect_d)) > 1e-3)
        {
            Vector3_t isect_d_cross_d_alpha = isect_d.cross(d_alpha);
            Vector3_t vp_a = (1.0/isect_d.cross(d_alpha).squaredNorm()) * (isect_d * (k.dot(isect_d_cross_d_alpha)) - d_alpha * (isect_k.dot(isect_d_cross_d_alpha)));
            va = (q - vp_a).norm();
        }

        number_t vb = std::numeric_limits<double>::infinity();
        if(1 - std::abs(d_beta.dot(isect_d)) > 1e-3)
        {
            Vector3_t isect_d_cross_d_beta = isect_d.cross(d_beta);
            Vector3_t vp_b = (1.0/isect_d.cross(d_beta).squaredNorm()) * (isect_d * (k.dot(isect_d_cross_d_beta)) - d_beta * (isect_k.dot(isect_d_cross_d_beta)));
            vb = (q - vp_b).norm();
        }

        auto v = std::min(va, vb);
        if(std::isinf(v))
        {
            return std::numeric_limits<double>::infinity();
        }

        return v;
    }
};

class FixedMomentMinSegmentDist
{
private:
    using number_t = double;
    using Vector3_t = Eigen::Vector3d;
    using Vector2_t = Eigen::Vector2d;
    using Matrix3_t = Eigen::Matrix3d;

    Vector3_t q;
    Vector3_t h1;
    Vector3_t h2;
    float t1;
    float t2;
public:
    FixedMomentMinSegmentDist(Vector3_t q, Vector3_t dirLowerBound, Vector3_t dirUpperBound, float t1, float t2)
            : q(std::move(q)), h1(std::move(dirLowerBound)), h2(std::move(dirUpperBound)), t1(t1), t2(t2) {}

    double operator()(const Vector& x, Vector& grad)
    {
        number_t phi_k = x[0];
        number_t theta_k = x[1];
        number_t r_k = x[2];

        // f(k)

        Vector3_t k = spherical2cart(phi_k, theta_k, r_k);
        Vector3_t kd = k / r_k;
        Vector3_t qp = q - kd*(q.dot(kd));
        auto qp_norm = qp.norm();
        Vector3_t qpd = qp / qp_norm;
        auto u = (qp - q).norm();

        auto calc_dist_for_gamma = [](const Vector3_t& qp, const Vector3_t& qpd, const Vector3_t& k, const Vector3_t& kd,
                const Vector3_t& h1, const Vector3_t& h2, number_t orientation, number_t sin_gamma, number_t t1, number_t t2)
        {
            auto cos_gamma = std::sqrt(1 - sin_gamma * sin_gamma);
            Vector3_t da = orientation * qpd * cos_gamma + kd.cross(qpd) * sin_gamma;
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

            //auto v = (qp.cross(d_alpha) - k).norm();
            Vector3_t p = d_alpha.cross(k);
            auto t = std::min(std::max(d_alpha.dot(qp), t1), t2);
            auto v = ((p + t*d_alpha) - qp).norm();
            return v;
        };

        auto t1_k_hypo_sqr = t1*t1 + r_k*r_k;
        auto t2_k_hypo_sqr = t2*t2 + r_k*r_k;
        number_t v;
        if(t1 >= 0 && t2 >= 0)
        {
            number_t sin_gamma;
            if(qp_norm > t2_k_hypo_sqr)
            {
                sin_gamma = std::min(r_k/std::sqrt(t2_k_hypo_sqr), (number_t)1.0f);
            }
            else if(qp_norm < t1_k_hypo_sqr)
            {
                sin_gamma = std::min(r_k/std::sqrt(t1_k_hypo_sqr), (number_t)1.0f);
            }
            else
            {
                sin_gamma = std::min(r_k/qp_norm, (number_t)1.0f);
            }
            v = calc_dist_for_gamma(qp, qpd, k, kd, h1, h2, 1, sin_gamma, t1, t2);
        }
        else if(t1 <= 0 && t2 <= 0)
        {
            number_t sin_gamma;
            if(qp_norm > t1_k_hypo_sqr)
            {
                sin_gamma = std::min(r_k/std::sqrt(t1_k_hypo_sqr), (number_t)1.0f);
            }
            else if(qp_norm < t2_k_hypo_sqr)
            {
                sin_gamma = std::min(r_k/std::sqrt(t2_k_hypo_sqr), (number_t)1.0f);
            }
            else
            {
                sin_gamma = std::min(r_k/qp_norm, (number_t)1.0f);
            }
            v = calc_dist_for_gamma(qp, qpd, k, kd, h1, h2, -1, sin_gamma, t1, t2);
        }
        else
        {
            number_t sin_gamma_a;
            if(qp_norm > t2_k_hypo_sqr)
            {
                sin_gamma_a = std::min(r_k/std::sqrt(t2_k_hypo_sqr), (number_t)1.0f);
            }
            else
            {
                sin_gamma_a = std::min(r_k/qp_norm, (number_t)1.0f);
            }
            number_t va = calc_dist_for_gamma(qp, qpd, k, kd, h1, h2, 1, sin_gamma_a, t1, t2);

            number_t sin_gamma_b;
            if(qp_norm > t1_k_hypo_sqr)
            {
                sin_gamma_b = std::min(r_k/std::sqrt(t1_k_hypo_sqr), (number_t)1.0f);
            }
            else
            {
                sin_gamma_b = std::min(r_k/qp_norm, (number_t)1.0f);
            }
            number_t vb = calc_dist_for_gamma(qp, qpd, k, kd, h1, h2, -1, sin_gamma_b, t1, t2);

            v = std::min(va, vb);
        }

        auto f = std::sqrt(u*u + v*v);
        return f;
    }
};

namespace pluckertree
{
    double FindMinDist(
            const Eigen::Vector3f& point,
            const Eigen::Vector3f& dirLowerBound,
            const Eigen::Vector3f& dirUpperBound,
            const Eigen::Vector3f& momentLowerBound,
            const Eigen::Vector3f& momentUpperBound,
            Eigen::Vector3f& minimum
    )
    {
        auto minimize = [point,dirLowerBound,dirUpperBound,momentLowerBound,momentUpperBound](Eigen::Vector3f& minimum){
            //nlopt::opt opt(nlopt::LN_COBYLA, 3);//LN_NELDERMEAD, LN_SBPLX
            nlopt::opt opt(nlopt::LN_SBPLX, 3);//LN_NELDERMEAD, LN_SBPLX

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
                return result;
            };
            opt.set_min_objective(obj_func, &fun);

            opt.set_xtol_rel(1e-3);
            opt.set_stopval(1e-3);
            opt.set_maxtime(1);
            opt.set_maxeval(1000);

            //Vector vec = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();
            Vector vec = minimum.cast<double>();
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
        };
        Eigen::Vector3f minHint;
        //minHint = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f);
        double minVal = 1E99;
        minHint = momentLowerBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; } //TODO: this can be higher if the parent node has a higher min dist value
        minHint = momentUpperBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        return minVal;

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
        auto minimize = [point,point_normal,dirLowerBound,dirUpperBound,momentLowerBound,momentUpperBound](Eigen::Vector3f& minimum){
            //nlopt::opt opt(nlopt::LN_COBYLA, 3);//LN_NELDERMEAD, LN_SBPLX
            nlopt::opt opt(nlopt::LN_SBPLX, 3);//LN_NELDERMEAD, LN_SBPLX

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
                return result;
            };
            opt.set_min_objective(obj_func, &fun);

            opt.set_xtol_rel(1e-3);
            opt.set_stopval(1e-3);
            opt.set_maxtime(1);
            opt.set_maxeval(1000);

            //Vector vec = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();
            Vector vec = minimum.cast<double>();
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
        };
        Eigen::Vector3f minHint;
        //minHint = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f);
        double minVal = 1E99;
        minHint = momentLowerBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; } //TODO: this can be higher if the parent node has a higher min dist value
        minHint = momentUpperBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        return minVal;
    }

    // Segments
    double segments::FindMinDist(
            const Eigen::Vector3f& point,
            const Eigen::Vector3f& dirLowerBound,
            const Eigen::Vector3f& dirUpperBound,
            const Eigen::Vector3f& momentLowerBound,
            const Eigen::Vector3f& momentUpperBound,
            float t1Min,
            float t2Max,
            Eigen::Vector3f& min
    )
    {
        auto minimize = [point,dirLowerBound,dirUpperBound,momentLowerBound,momentUpperBound, t1Min, t2Max](Eigen::Vector3f& minimum){
            nlopt::opt opt(nlopt::LN_COBYLA, 3);//LN_NELDERMEAD, LN_SBPLX

            std::vector<double> lb(momentLowerBound.data(), momentLowerBound.data() + momentLowerBound.rows() * momentLowerBound.cols());
            opt.set_lower_bounds(lb);

            std::vector<double> hb(momentUpperBound.data(), momentUpperBound.data() + momentUpperBound.rows() * momentUpperBound.cols());
            opt.set_upper_bounds(hb);

            FixedMomentMinSegmentDist fun(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>(), t1Min, t2Max);
            auto obj_func = [](const std::vector<double> &x, std::vector<double> &grad, void* f_data) -> double {
                FixedMomentMinSegmentDist* fun = reinterpret_cast<FixedMomentMinSegmentDist*>(f_data);
                Vector x_vect = Eigen::Vector3d {x[0], x[1], x[2]};
                Vector grad_vect = Eigen::Vector3d {0, 0, 0};
                auto result = (*fun)(x_vect, grad_vect);
                return result;
            };
            opt.set_min_objective(obj_func, &fun);

            opt.set_xtol_rel(1e-3);
            opt.set_stopval(1e-3);
            opt.set_maxtime(1);
            opt.set_maxeval(1000);

            //Vector vec = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f).cast<double>();
            Vector vec = minimum.cast<double>();
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
        };
        Eigen::Vector3f minHint;
        //minHint = (momentLowerBound + (momentUpperBound - momentLowerBound)/2.0f);
        double minVal = 1E99;
        minHint = momentLowerBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; } //TODO: this can be higher if the parent node has a higher min dist value
        minHint = momentUpperBound;
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentLowerBound.x(), momentUpperBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentUpperBound.y(), momentLowerBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        if(minVal < 1e-3){ return minVal; }
        minHint = Eigen::Vector3f(momentUpperBound.x(), momentLowerBound.y(), momentUpperBound.z());
        minVal = std::min(minVal, minimize(minHint));
        Diag::minimizations++;
        return minVal;
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
thread_local unsigned int pluckertree::Diag::visited = 0;
thread_local unsigned int pluckertree::Diag::minimizations = 0;
std::optional<std::function<void(float, float, float, int)>> pluckertree::Diag::on_node_visited;
std::optional<std::function<void(float, float, int)>> pluckertree::Diag::on_node_enter;
std::optional<std::function<void(float, float, float, int)>> pluckertree::Diag::on_node_leave;
std::optional<std::function<void(float, float, float, float)>> pluckertree::Diag::on_build_variance_calculated;
bool pluckertree::Diag::force_visit_all = false;

std::random_device pluckertree::MyRand::rand_dev;

struct GridPoint
{
    Eigen::Vector3f pos;
    Eigen::Vector3f grad;
    float dist;
};

/*std::vector<GridPoint> CalculateGrid(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& query_point,
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
    FixedMomentMinHitDist f(point.cast<double>(), query_point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

    auto x_steps = int((x_end - x_start) / resolution);
    auto y_steps = int((y_end - y_start) / resolution);
    auto z_steps = int((z_end - z_start) / resolution);

    std::vector<GridPoint> points(x_steps * y_steps * z_steps);

    for(int x_i = 0; x_i < x_steps; x_i++)
    {
        std::cout << ((float)x_i/(float)x_steps)*100.0f << "%\r";
        std::cout.flush();
        auto x = x_start + resolution*x_i;
        for(int y_i = 0; y_i < y_steps; y_i++)
        {
            auto y = y_start + resolution*y_i;
            for(int z_i = 0; z_i < z_steps; z_i++)
            {
                auto z = z_start + resolution*z_i;

                Vector vect = cart2spherical(Eigen::Vector3d(x, y, z));
                Vector grad;
                auto dist = f(vect, grad);

                auto& curPoint = points[(x_i * y_steps * z_steps) + (y_i * z_steps) + z_i];
                curPoint.pos = Eigen::Vector3f(x, y, z);
                curPoint.dist = dist;
                //curPoint.grad = grad.cast<float>();
            }
        }
    }
    std::cout << std::endl;

    return points;
}*/

std::vector<GridPoint> CalculateBallSlice(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& dirLowerBound,
        const Eigen::Vector3f& dirUpperBound,
        const Eigen::Vector3f& mlb,
        const Eigen::Vector3f& mub,
        float resolution
)
{
    FixedMomentMinDist f(point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

    Eigen::Vector3f stepSize = (mub - mlb) / resolution;

    std::vector<GridPoint> points(resolution * resolution * resolution);

    for(int x_i = 0; x_i < resolution; x_i++)
    {
        std::cout << ((float)x_i/(float)resolution)*100.0f << "%\r";
        std::cout.flush();
        auto x = mlb.x() + stepSize.x()*x_i;
        for(int y_i = 0; y_i < resolution; y_i++)
        {
            auto y = mlb.y() + stepSize.y()*y_i;
            for(int z_i = 0; z_i < resolution; z_i++)
            {
                auto z = mlb.z() + stepSize.z()*z_i;

                Vector vect = Eigen::Vector3d(x, y, z);
                Vector grad;
                auto dist = f(vect, grad);

                auto& curPoint = points[(x_i * resolution * resolution) + (y_i * resolution) + z_i];
                curPoint.pos = spherical2cart(x, y, z);
                curPoint.dist = dist;
                //curPoint.grad = grad.cast<float>();
            }
        }
    }
    std::cout << std::endl;

    return points;
}

std::vector<GridPoint> CalculateBallSlice(
        const Eigen::Vector3f& point,
        const Eigen::Vector3f& query_point,
        const Eigen::Vector3f& dirLowerBound,
        const Eigen::Vector3f& dirUpperBound,
        const Eigen::Vector3f& mlb,
        const Eigen::Vector3f& mub,
        float resolution
)
{
    FixedMomentMinHitDist f(point.cast<double>(), query_point.cast<double>(), dirLowerBound.cast<double>(), dirUpperBound.cast<double>());

    Eigen::Vector3f stepSize = (mub - mlb) / resolution;

    std::vector<GridPoint> points(resolution * resolution * resolution);

    for(int x_i = 0; x_i < resolution; x_i++)
    {
        std::cout << ((float)x_i/(float)resolution)*100.0f << "%\r";
        std::cout.flush();
        auto x = mlb.x() + stepSize.x()*x_i;
        for(int y_i = 0; y_i < resolution; y_i++)
        {
            auto y = mlb.y() + stepSize.y()*y_i;
            for(int z_i = 0; z_i < resolution; z_i++)
            {
                auto z = mlb.z() + stepSize.z()*z_i;

                Vector vect = Eigen::Vector3d(x, y, z);
                Vector grad;
                auto dist = f(vect, grad);

                auto& curPoint = points[(x_i * resolution * resolution) + (y_i * resolution) + z_i];
                curPoint.pos = spherical2cart(x, y, z);
                curPoint.dist = dist;
                //curPoint.grad = grad.cast<float>();
            }
        }
    }
    std::cout << std::endl;

    return points;
}

void show_me_the_grid(std::string& file,
                      const Eigen::Vector3f& dlb,
                      const Eigen::Vector3f& dub,
                      const Eigen::Vector3f& mlb,
                      const Eigen::Vector3f& mub,
                      const Eigen::Vector3f& q)
{
    /*Eigen::Vector3f dlb = Eigen::Vector3f(0,0,-1);
    Eigen::Vector3f dub = Eigen::Vector3f(-std::sqrt(2)/2.0, std::sqrt(2)/2, 0);
    Eigen::Vector3f mlb(-M_PI, 0.785398185, 1);
    Eigen::Vector3f mub(-M_PI/2, 2.3561945, 80);
    Eigen::Vector3f q(25.216011, 86.2393799, 64.2581253);
    Eigen::Vector3f q_normal(0.742901862, 0.662636876, 0.0949169919);*/

    float resolution = 300;
    //auto data = CalculateBallSlice(q, q_normal, dlb, dub, mlb, mub, resolution);
    auto data = CalculateBallSlice(q, dlb, dub, mlb, mub, resolution);

    std::fstream myfile;
    myfile = std::fstream(file, std::ios::out | std::ios::binary);

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
    myfile.close();
}

/////

}