use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U6},
    storage::Owned,
    DVector, MatrixMN, Unit, VecStorage, Vector3, Vector6,
};
use cv_core::{Bearing, FeatureMatch, Pose, RelativeCameraPose, Skew3, TriangulatorRelative};
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem};

#[derive(Clone)]
pub struct TwoViewOptimizer<I, T> {
    matches: I,
    pub pose: RelativeCameraPose,
    triangulator: T,
    residuals: DVector<f64>,
    jacobian: MatrixMN<f64, Dynamic, U6>,
}

impl<I, P, T> TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative,
{
    pub fn new(matches: I, pose: RelativeCameraPose, triangulator: T) -> Self {
        let (residuals, jacobian) =
            Self::compute_residuals_and_jacobian(matches.clone(), pose, &triangulator);
        Self {
            matches,
            pose,
            triangulator,
            residuals,
            jacobian,
        }
    }

    fn compute_residuals_and_jacobian(
        matches: I,
        pose: RelativeCameraPose,
        triangulator: &T,
    ) -> (DVector<f64>, MatrixMN<f64, Dynamic, U6>) {
        // Initialize the jacobian with all zeros.
        let residual_count = matches.clone().count() * 3;
        let mut jacobian = MatrixMN::zeros_generic(Dynamic::new(residual_count), U6);
        let mut residuals = DVector::zeros(residual_count);

        // Loop through every match and row zipped together.
        for (ix, FeatureMatch(a, b)) in matches.enumerate() {
            let a = a.bearing();
            let b = b.bearing();
            let cam_a_point = if let Some(point) = triangulator.triangulate_relative(pose, a, b) {
                point
            } else {
                continue;
            };
            let (cam_b_point, pose_jacobian_a_b) = pose.transform_jacobian_pose(cam_a_point);

            // Get the point on the bearing `b` closest to the point.
            let cam_b_p_hat = cam_b_point.coords.dot(&b) * b.into_inner();
            let cam_a_p_hat = cam_a_point.coords.dot(&a) * a.into_inner();

            // Compute the residual vector.
            let residual_vector = cam_b_p_hat - cam_b_point.coords;

            // The residual is the distance to the triangulated point from the projected point.
            residuals[ix * 3] = residual_vector.x;
            residuals[ix * 3 + 1] = residual_vector.y;
            residuals[ix * 3 + 2] = residual_vector.z;
            jacobian
                .row_mut(ix * 3)
                .copy_from(&(pose_jacobian_a_b * -Vector3::x()).transpose());
            jacobian
                .row_mut(ix * 3 + 1)
                .copy_from(&(pose_jacobian_a_b * -Vector3::y()).transpose());
            jacobian
                .row_mut(ix * 3 + 2)
                .copy_from(&(pose_jacobian_a_b * -Vector3::z()).transpose());
        }

        (residuals, jacobian)
    }
}

impl<I, P, T> LeastSquaresProblem<f64, Dynamic, U6> for TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative + Clone,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, U6>;
    type ParameterStorage = Owned<f64, U6>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector6<f64>) {
        self.pose.translation.vector = x.xyz();
        let x = x.as_slice();
        self.pose.rotation = Skew3(Vector3::new(x[3], x[4], x[5])).into();

        // Clear out the old residuals and jacobian first to reduce maximum memory consumption.
        self.residuals = DVector::zeros(0);
        self.jacobian = MatrixMN::zeros_generic(Dynamic::new(0), U6);

        // Compute the new residuals and jacobian.
        let (residuals, jacobian) = Self::compute_residuals_and_jacobian(
            self.matches.clone(),
            self.pose,
            &self.triangulator,
        );

        self.residuals = residuals;
        self.jacobian = jacobian;
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> Vector6<f64> {
        let skew: Skew3 = self.pose.rotation.into();
        if let [x, y, z] = *skew.as_slice() {
            self.pose.translation.vector.push(x).push(y).push(z)
        } else {
            unreachable!()
        }
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(self.residuals.clone())
    }

    // /// Compute the Jacobian of the residual vector.
    // fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U6>> {
    //     let mut clone = self.clone();
    //     differentiate_numerically(&mut clone)
    // }

    /// Compute the Jacobian of the pose.
    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U6>> {
        Some(self.jacobian.clone())
    }
}
