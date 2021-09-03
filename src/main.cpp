#include <iostream>
#include <Eigen/Core>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <igl/readOBJ.h>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <map>
#include <limits>

class UnionFind
{
public:
    UnionFind(int n)
    {
        parents.resize(n);
        for (int i = 0; i < n; i++)
            parents[i] = i;
    }

    int find(int k)
    {
        if (k != parents[k])
        {
            int ans = find(parents[k]);
            parents[k] = ans;
        }
        return parents[k];
    }

    void takeUnion(int k1, int k2)
    {
        int root1 = find(k1);
        int root2 = find(k2);
        if (root1 != root2)
        {
            parents[root1] = root2;
        }
    }

private:
    std::vector<int> parents;
};

void findBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd& independent, Eigen::SparseMatrix<double>& P, Eigen::VectorXd &b)
{
    int nverts = V.rows();

    // floating-point computation tolerance
    double eps = 1e-6;

    std::vector<std::vector<int> > bdryL;
    igl::boundary_loop(F, bdryL);

    double mins[3];
    double maxs[3];
    for (int i = 0; i < 3; i++)
    {
        mins[i] = std::numeric_limits<double>::infinity();
        maxs[i] = -std::numeric_limits<double>::infinity();
    }

    std::vector<int> allbdry;

    for (auto& L : bdryL)
    {
        for (auto it : L)
        {
            allbdry.push_back(it);
            for (int i = 0; i < 3; i++)
            {
                mins[i] = std::min(mins[i], V(it, i));
                maxs[i] = std::max(maxs[i], V(it, i));
            }
        }
    }

    UnionFind uf(nverts);

    for (int i = 0; i < allbdry.size(); i++)
    {
        bool onboundary[3];
        Eigen::Vector3d pos = V.row(allbdry[i]).transpose();

        for (int j = 0; j < 3; j++)
        {
            onboundary[j] = (std::fabs(pos[j] - mins[j]) < eps) || (std::fabs(pos[j] - maxs[j]) < eps);
        }

        for (int j = 0; j < i; j++)
        {
            bool ok = true;
            Eigen::Vector3d pos2 = V.row(allbdry[j]).transpose();
            for (int k = 0; k < 3; k++)
            {
                if (onboundary[k])
                {
                    if (!((std::fabs(pos2[k] - mins[k]) < eps) || (std::fabs(pos2[k] - maxs[k]) < eps)))
                        ok = false;
                }
                else
                {
                    if (!(std::fabs(pos[k] - pos2[k]) < eps))
                        ok = false;
                }
            }

            if (ok)
            {
                uf.takeUnion(allbdry[i], allbdry[j]);
            }
        }
    }

    std::vector<int> counts(nverts);

    for (int i = 0; i < nverts; i++)
    {
        counts[uf.find(i)]++;
    }

    std::vector<int> independentverts;
    std::map<int, int> invmap;

    for (int i = 0; i < nverts; i++)
    {
        if (i == uf.find(i))
        {
            invmap[i] = independentverts.size();
            independentverts.push_back(i);
        }
    }

    int nind = independentverts.size();

    independent.resize(3 * nind);
    for (int i = 0; i < nind; i++)
    {
        independent.segment<3>(3 * i) = V.row(independentverts[i]).transpose();
    }

    std::vector<Eigen::Triplet<double> > Pcoeffs;
    b.resize(3 * nverts);
    for (int i = 0; i < nverts; i++)
    {
        int ind = invmap[uf.find(i)];
        for (int j = 0; j < 3; j++)
        {
            Pcoeffs.push_back({ 3 * i + j, 3 * ind + j, 1.0 });
        }

        b.segment<3>(3 * i) = V.row(i).transpose() - independent.segment<3>(3 * ind);
    }

    P.resize(3 * nverts, 3 * nind);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());
}

/**
 * @brief cross matrix of a 3D vector 
 * @param v: a 3D vector
 * @return ret: cross matrix, a 3 x 3 matrix
 */
Eigen::Matrix3d crossMatrix(Eigen::Vector3d v)
{
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    return ret;
}

/**
 * @brief caculate the derivative and hessian of a triangle area 
 * @param q0, q1, q2: the three vertices of a triangle, 3D vectors
 * @return deriv, hess: derivative and hessian of this triangle, pass by reference
 * @return E: surface area of a triangle
 */
double triangleArea(
    Eigen::Vector3d q0,
    Eigen::Vector3d q1,
    Eigen::Vector3d q2,
    Eigen::Matrix<double, 1, 9>* deriv,
    Eigen::Matrix<double, 9, 9>* hess
)
{
    Eigen::Vector3d n = (q1 - q0).cross(q2 - q0);

    // E = surface area energy
    double E = 0.5 * n.norm();

    if (deriv || hess)
    {
        /**
         * Jn = Jacobian of surface area normal
         * Jn is a 3 x 9 dense matrix
         */
        Eigen::Matrix<double, 3, 9> Jn;
        Jn.block<3, 3>(0, 0) = crossMatrix(q2 - q1);
        Jn.block<3, 3>(0, 3) = crossMatrix(q0 - q2);
        Jn.block<3, 3>(0, 6) = crossMatrix(q1 - q0);

        if (deriv)
        {
            /**
             * J = Jacobian of surface area energy
             * J = 1/(2||n||) * n^T * Jn
             * J is a 1 x 9 dense matrix
             */
            *deriv = 0.5 / n.norm() * n.transpose() * Jn;
        }

        if (hess)
        {
            /**
             * B is a 9 x 9 dense matrix
             */
            Eigen::Matrix<double, 9, 9> B;
            B.setZero();
            B.block<3, 3>(0, 3) = -crossMatrix(n);
            B.block<3, 3>(0, 6) = crossMatrix(n);
            B.block<3, 3>(3, 0) = crossMatrix(n);
            B.block<3, 3>(3, 6) = -crossMatrix(n);
            B.block<3, 3>(6, 0) = -crossMatrix(n);
            B.block<3, 3>(6, 3) = crossMatrix(n);

            // I is a 3 x 3 identity matrix
            Eigen::Matrix3d I;
            I.setIdentity();

            /**
             * H = Hessian of surface area energy
             * H = 1/2 * (Jn)^T * (I/||n|| - nn^T/(||n||^3)) * Jn + 1/(2||n||) * B
             * H is a 9 x 9 dense matrix
             */
            *hess = 0.5 * Jn.transpose() * (I / n.norm() - n * n.transpose() / n.norm() / n.norm() / n.norm()) * Jn + 0.5 / n.norm() * B;
        }
    }

    return E;
}

/**
 * @brief map the derivative and hessian of one triangle to the the derivative and hessian of a mesh
 * @param V: vertices positions of a mesh
 * @param F: face/connectivity of a mesh
 * @return deriv, hess: derivative and hessian of a mesh, pass by reference
 * @return totarea: surface area of a mesh
 */
double surfaceArea(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double> >* hess)
{
    int nverts = V.rows();

    if (deriv)
    {
        deriv->resize(3 * nverts);
        deriv->setZero();
    }
    if (hess)
        hess->clear();

    Eigen::Matrix<double, 1, 9> aderiv;
    Eigen::Matrix<double, 9, 9> ahess;

    int nfaces = F.rows();
    double totarea = 0;

    for (int i = 0; i < nfaces; i++)
    {
        // extract one triangle from the mesh
        Eigen::Vector3i face = F.row(i).transpose();

        Eigen::Vector3d q0 = V.row(face[0]).transpose();
        Eigen::Vector3d q1 = V.row(face[1]).transpose();
        Eigen::Vector3d q2 = V.row(face[2]).transpose();

        // get derivative and hessian of this triangle
        double a = triangleArea(q0, q1, q2, deriv ? &aderiv : NULL, hess ? &ahess : NULL);
        totarea += a;

        // map derivative
        if (deriv)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    (*deriv)[3 * face[j] + k] += aderiv(0, 3 * j + k);
                }
            }
        }

        // map hessian
        if (hess)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int m = 0; m < 3; m++)
                        {
                            hess->push_back({ 3 * face[j] + k, 3 * face[l] + m, ahess(3 * j + k, 3 * l + m) });
                        }
                    }
                }
            }
        }
    }

    return totarea;
}

/**
 * @brief caculate the derivative and willmore energy of a triangle 
 * @param q0, q1, q2: the three vertices of a triangle, 3D vectors
 * @return deriv: derivative of this triangle's willmore energy, pass by reference
 * @return E: willmore energy of a triangle
 */
double triangleWillmore(
    Eigen::Vector3d q0,
    Eigen::Vector3d q1,
    Eigen::Vector3d q2,
    Eigen::Matrix<double, 1, 9>* deriv
)
{
    Eigen::Matrix<double, 1, 9> areaderiv;
    Eigen::Matrix<double, 9, 9> areahess;

    triangleArea(q0, q1, q2, &areaderiv, deriv ? &areahess : NULL);

    double E = areaderiv.squaredNorm();

    if (deriv)
    {
        // Jwillmore = 2 * J * H
        *deriv = 2.0 * areaderiv * areahess;
    }

    return E;
}

/**
 * @brief map the derivative and hessian of one triangle to the the derivative and hessian of a mesh
 * @param V: vertices positions of a mesh
 * @param F: face/connectivity of a mesh
 * @return deriv: derivative of a mesh's willmore energy, pass by reference
 * @return tot: total willmore energy of a mesh
 */
double willmore(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd* deriv)
{
    int nverts = V.rows();

    // if deriv not nullptr, reset deriv vector
    if (deriv)
    {
        deriv->resize(3 * nverts);
        deriv->setZero();
    }
    
    // init derivative and hessian of one triangle
    Eigen::Matrix<double, 1, 9> aderiv;
    Eigen::Matrix<double, 9, 9> ahess;

    int nfaces = F.rows();
    double tot = 0;

    for (int i = 0; i < nfaces; i++)
    {
        // extract one triangle from the mesh
        Eigen::Vector3i face = F.row(i).transpose();

        Eigen::Vector3d q0 = V.row(face[0]).transpose();
        Eigen::Vector3d q1 = V.row(face[1]).transpose();
        Eigen::Vector3d q2 = V.row(face[2]).transpose();

        // get derivative and willmore energy of this triangle
        double E = triangleWillmore(q0, q1, q2, deriv ? &aderiv : NULL);
        tot += E;

        // map derivative
        if (deriv)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    (*deriv)[3 * face[j] + k] += aderiv(0, 3 * j + k);
                }
            }
        }
    }

    return tot;
}

/**
 * @brief newton's method solver
 * @param independent: indepedent set of vertices positions on a mesh
 * @param F: face/connectivity of a mesh
 * @param P: selection matrix of indepedent set
 * @param b: boundary vertices of a mesh
 * @return newindependent: updated indepedent set of vertices positions, pass by reference
 * @return gradient, descentDir: graident and descent direction (per vertex) of a mesh, pass by reference
 */
void newton(const Eigen::VectorXd &independent, const Eigen::MatrixXi& F, const Eigen::SparseMatrix<double> &P, const Eigen::VectorXd &b, Eigen::VectorXd &newindependent, Eigen::MatrixXd &gradient, Eigen::MatrixXd &descentDir)
{
    Eigen::VectorXd flatV = P * independent + b;
    int nverts = flatV.size() / 3;
    Eigen::MatrixXd V(nverts, 3);
    for (int i = 0; i < nverts; i++)
    {
        V.row(i) = flatV.segment<3>(3 * i).transpose();
    }

    Eigen::VectorXd deriv;
    std::vector<Eigen::Triplet<double> > hesscoeffs;
    double oldenergy = willmore(V, F, &deriv);  // note that we do not need hessian for willmore energy
    
    Eigen::SparseMatrix<double> L(nverts, nverts);
    Eigen::SparseMatrix<double> M(nverts, nverts);
    // L = stiffness matrix/discrete laplacian of a mesh
    igl::cotmatrix(V, F, L);
    // M = mass matrix of a mesh, computed with barycentric coords
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

    std::vector<Eigen::Triplet<double> > bigLcoeffs;
    for (int k=0; k<L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
        {
            for (int j = 0; j < 3; j++)
            {
                bigLcoeffs.push_back({ 3 * (int)it.row() + j, 3 * (int)it.col() + j, it.value() });
            }
        }

    std::vector<Eigen::Triplet<double> > bigMcoeffs;
    for (int i = 0; i < nverts; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            bigMcoeffs.push_back({ 3 * i + j, 3 * i + j, M.coeff(i,i) });
        }
    }

    Eigen::SparseMatrix<double> bigL(3 * nverts, 3 * nverts);
    bigL.setFromTriplets(bigLcoeffs.begin(), bigLcoeffs.end());

    Eigen::SparseMatrix<double> bigM(3 * nverts, 3 * nverts);
    bigM.setFromTriplets(bigMcoeffs.begin(), bigMcoeffs.end());
    Eigen::SparseMatrix<double> projM = P.transpose() * bigM * P;

    Eigen::SparseMatrix<double> projL = P.transpose() * bigL.transpose() * P;
    // init solver to solve projL*y = -projJ for y and projL*dx = projM*y for dx
    Eigen::SPQR<Eigen::SparseMatrix<double> > Lsolver(projL);
    Eigen::VectorXd projJ = P.transpose() * deriv;
    Eigen::VectorXd rhs = -projJ;
    Eigen::VectorXd y = Lsolver.solve(rhs);
    Eigen::VectorXd rhs2 = projM * y;
    Eigen::VectorXd dx = Lsolver.solve(rhs2);

    // compute the residuals and print out
    double resid1 = (projL * dx - rhs2).norm();
    double resid2 = (projL * y - rhs).norm();

    std::cout << std::endl;
    std::cout << "Solver residuals: " << resid1 << " " << resid2 << std::endl;

    // backtracking line search
    Eigen::VectorXd deltaV = P * dx;

    double step = 1.0;
    while (true)
    {
        Eigen::MatrixXd newV = V;
        for (int i = 0; i < nverts; i++)
            newV.row(i) += step * deltaV.segment<3>(3 * i);
        // compute the updated energy with the new vertices
        double newenergy = willmore(newV, F, NULL);
        // if the updated energy is lower, break the loop
        if (newenergy <= oldenergy)
        {
            std::cout << "Accepted step " << step << std::endl;
            std::cout << "Old Willmore energy: " << oldenergy << ", new energy: " << newenergy << std::endl;
            Eigen::VectorXd unprojJ = P * projJ;
            gradient.resize( nverts, 3);
            descentDir.resize(nverts, 3);
            newindependent = independent + step * dx;
            for (int i = 0; i < nverts; i++)
            {
                gradient.row(i) = unprojJ.segment<3>(3 * i);
                descentDir.row(i) = deltaV.segment<3>(3 * i);
            }
            return;
        }
        // if the current step is too aggressive, halve the step and retry
        step /= 2;
    }
}

/**
 * @brief main function
 */
int main(int argc, char *argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    std::string objfile = "cube.obj";

    // find obj file in current folder and its parent folder
    // load vertices (V) and faces (F)
    Eigen::MatrixXd values;
    if (!igl::readOBJ(objfile, V, F))
    {
        if (!igl::readOBJ("../" + objfile, V, F))
        {
            std::cerr << "error reading the mesh file" << std::endl;
            return -1;
        }
    }

    int nverts = V.rows();
    Eigen::VectorXd independent;
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd b;
    findBoundaries(V, F, independent, P, b);

    Eigen::MatrixXd noise = independent;
    noise.setRandom();
    double noisemag = 1e-4;
    independent = independent + noisemag * noise;
    
    // test mesh set-up
    /*V.resize(9,3);
    V << -1, -1, 0,
        0, -1, 0,
        1, -1, 0,
        -1, 0, 0,
        0, 0, 0,
        1, 0, 0,
        -1, 0, -1,
        0, 0, -1,
        1, 0, -1;

    F.resize(8, 3);
    F << 0, 1, 3,
        3, 1, 4,
        1, 2, 4,
        4, 2, 5,
        3, 4, 6,
        6, 4, 7,
        4, 5, 7,
        7, 5, 8;
      */  

    polyscope::init();

    Eigen::VectorXd newindependent;
    Eigen::MatrixXd gradient;
    Eigen::MatrixXd descentDir;
    // add reference/initial mesh to viewer
    auto *prevMesh = polyscope::registerSurfaceMesh("Mesh 0", V, F);
    
    // solver iteration steps setting
    int noutersteps = 1;    // number of minimal surface optimization steps
    int ninnersteps = 100;  // number of newton's method steps

    for (int outer = 0; outer < noutersteps; outer++)
    {
        for (int inner = 0; inner < ninnersteps; inner++)
        {
            newton(independent, F, P, b, newindependent, gradient, descentDir);
            independent = newindependent;
        }
        Eigen::VectorXd newVflat = P * independent + b;
        Eigen::MatrixXd newV(nverts, 3);
        for (int i = 0; i < nverts; i++)
            newV.row(i) = newVflat.segment<3>(3 * i);

        // add the mesh at this optimization step tp viewer
        prevMesh = polyscope::registerSurfaceMesh("Mesh " + std::to_string(outer+1), newV, F);
        prevMesh->addVertexVectorQuantity("Gradient", gradient);
        prevMesh->addVertexVectorQuantity("Descent Dir", descentDir);        
    }
    
    // visualize
    polyscope::show();
}
