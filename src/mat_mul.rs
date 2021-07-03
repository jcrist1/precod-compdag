use num_traits::One;
use num_traits::Zero;
use rand;
use rand_distr;
use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::SubAssign;
use thiserror::Error;

/// This module is for basic matrix math. Thanks to rust operator overloading I can still get semi
/// ergonomic linear algebra routines without too much trouble. The whole idea is that we want the
/// compiler to check the veracity of the computational graph, so static sizes for the different
/// layers/nodes.
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("Unable to access access {index:?} from axis withe {max:?} elements")]
    AxisAccessError { max: usize, index: usize },
}

/// Vector is a statically sized vector structure which is stored on the heap.
/// Apparently, in the words of the great Oriol Muñoz "heaps are small". The vector itself doesn't
/// gain much by being heap allocated, but at least we shouldn't be moving the data between
/// different stacks if we're passing them around as messages in threads... which we are
///
/// It would be better to implement this as a statically sized array, whose lookup is defined by
/// the const generic patterns, instead of nested arrays. But I don't want to dig into alloc yet.
#[derive(Debug, Clone)]
pub struct Vector<F, const N: usize> {
    pub(crate) data: Box<[F; N]>,
}

impl<F, const N: usize> Vector<F, N>
where
    F: Copy,
{
    pub fn set(&mut self, index: usize, value: &F) -> Result<(), MatrixError> {
        if index >= N {
            Err(MatrixError::AxisAccessError { max: N, index })
        } else {
            self.data[index] = *value;
            Ok(())
        }
    }

    pub fn get(&self, index: usize) -> Result<&F, MatrixError> {
        if index >= N {
            Err(MatrixError::AxisAccessError { max: N, index })
        } else {
            Ok(&self.data[index])
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Result<&mut F, MatrixError> {
        if index >= N {
            Err(MatrixError::AxisAccessError { max: N, index })
        } else {
            Ok(&mut self.data[index])
        }
    }

    pub fn from_distribution<Rng, Distribution>(rng: &mut Rng, dist: &Distribution) -> Self
    where
        Rng: rand::Rng,
        Distribution: rand_distr::Distribution<F>,
    {
        let data = [(); N].map(|_| dist.sample(rng));
        Vector {
            data: Box::new(data),
        }
    }

    /// This allocates a new Vector
    pub fn map<U, FnFToU: FnMut(&F) -> U>(&self, function: FnFToU) -> Vector<U, N> {
        let mut iter = self.data.iter().map(function);
        let data = Box::new([(); N].map(|_| iter.next().unwrap()));
        Vector { data }
    }

    /// This maps the existing data in place
    pub fn map_inplace<FnFToF: FnMut(&F) -> F>(&mut self, mut function: FnFToF) -> () {
        self.data.iter_mut().for_each(|x| *x = function(x));
    }

    /// I don't know that this should ever be used, but maybe?
    pub fn map_consume<U, FnFToU: FnMut(F) -> U>(self, function: FnFToU) -> Vector<U, N> {
        Vector {
            data: Box::new(self.data.map(function)),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.data.iter_mut()
    }
}

impl<F, const N: usize> Vector<F, N>
where
    F: Zero + Copy,
{
    pub fn new() -> Self {
        let data = Box::new([F::zero(); N]);
        Vector { data }
    }

    /// If the iterator is spent, then we default to F::zero()
    pub fn from_iter<I: Iterator<Item = F>>(mut iter: I) -> Self {
        let data = Box::new([(); N].map(|_| iter.next().unwrap_or(F::zero())));
        Vector { data }
    }
}

impl<F, const N: usize> Vector<F, N>
where
    F: One + Copy,
{
    pub fn ones() -> Self {
        let data = Box::new([F::one(); N]);
        Vector { data }
    }
}

impl<F, const N: usize> Vector<F, N>
where
    F: Mul<F, Output = F> + Copy + Zero,
{
    // Todo Zero is technically not needed if we don't use the from_iter impl
    pub fn component_mul(&self, other: &Vector<F, N>) -> Vector<F, N> {
        Vector::from_iter(self.iter().zip(other.iter()).map(|(x, y)| *x * *y))
    }
}

impl<'a, F, const N: usize> Add<&'a Vector<F, N>> for &'a Vector<F, N>
where
    F: Add<F, Output = F> + Zero + Copy,
{
    type Output = Vector<F, N>;

    fn add(self, rhs: &'a Vector<F, N>) -> Self::Output {
        let mut new_data = [F::zero(); N];
        self.data
            .iter()
            .zip(rhs.data.iter())
            .zip(new_data.iter_mut())
            .for_each(|((left, right), new)| *new = *left + *right);

        Vector {
            data: Box::new(new_data),
        }
    }
}

impl<'a, F, const N: usize> Add<Vector<F, N>> for &'a Vector<F, N>
where
    F: Add<F, Output = F> + Zero + Copy,
{
    type Output = Vector<F, N>;

    fn add(self, rhs: Vector<F, N>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, F, const N: usize> Add<&'a Vector<F, N>> for Vector<F, N>
where
    F: Add<F, Output = F> + Zero + Copy,
{
    type Output = Vector<F, N>;

    fn add(self, rhs: &'a Vector<F, N>) -> Self::Output {
        &self + rhs
    }
}

impl<F, const N: usize> Add<Vector<F, N>> for Vector<F, N>
where
    F: Add<F, Output = F> + Zero + Copy,
{
    type Output = Vector<F, N>;

    fn add(self, rhs: Vector<F, N>) -> Self::Output {
        &self + &rhs
    }
}

/// We're going with multiplication being matrix multiplication
impl<'a, F, const N: usize> Mul<&'a Vector<F, N>> for &'a Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = F;

    fn mul(self, rhs: &'a Vector<F, N>) -> F {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .fold(F::zero(), |accum, (x, y)| accum + *x * *y)
    }
}

impl<'a, F, const N: usize> Mul<&'a Vector<F, N>> for Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = F;

    fn mul(self, rhs: &'a Vector<F, N>) -> F {
        &self * rhs
    }
}

impl<'a, F, const N: usize> Mul<Vector<F, N>> for &'a Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = F;

    fn mul(self, rhs: Vector<F, N>) -> F {
        self * &rhs
    }
}

impl<F, const N: usize> Mul<Vector<F, N>> for Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = F;

    fn mul(self, rhs: Vector<F, N>) -> F {
        &self * &rhs
    }
}

impl<'a, F, const N: usize> AddAssign<&'a Vector<F, N>> for Vector<F, N>
where
    F: AddAssign<F> + Copy,
{
    fn add_assign(&mut self, rhs: &'a Vector<F, N>) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x += *y)
    }
}

impl<F, const N: usize> AddAssign<Vector<F, N>> for Vector<F, N>
where
    F: AddAssign<F> + Copy,
{
    fn add_assign(&mut self, rhs: Vector<F, N>) {
        *self += &rhs
    }
}

impl<'a, F, const N: usize> SubAssign<&'a Vector<F, N>> for Vector<F, N>
where
    F: SubAssign<F> + Copy,
{
    fn sub_assign(&mut self, rhs: &'a Vector<F, N>) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x -= *y)
    }
}

impl<F, const N: usize> SubAssign<Vector<F, N>> for Vector<F, N>
where
    F: SubAssign<F> + Copy,
{
    fn sub_assign(&mut self, rhs: Vector<F, N>) {
        *self -= &rhs
    }
}

impl<F, const N: usize> PartialEq<Vector<F, N>> for Vector<F, N>
where
    F: PartialEq + Copy,
{
    fn eq(&self, rhs: &Vector<F, N>) -> bool {
        self.iter().zip(rhs.iter()).all(|(x, y)| *x == *y)
    }
}

impl<F, const N: usize> Eq for Vector<F, N> where F: Eq + Copy {}

/// I will probably move AddAssign<F>, to some kind of broadcast trait, when I'm ready to use macros.
/// Vectors are practically defined by their abelian + and scalar multiplication so I will keep
/// that here.
impl<'a, F, const N: usize> Mul<F> for &'a Vector<F, N>
where
    F: Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: F) -> Self::Output {
        self.map(|x| *x * rhs)
    }
}

impl<F, const N: usize> Mul<F> for Vector<F, N>
where
    F: Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: F) -> Self::Output {
        &self * rhs
    }
}

impl<F, const N: usize> AddAssign<F> for Vector<F, N>
where
    F: AddAssign<F> + Copy,
{
    fn add_assign(&mut self, rhs: F) {
        self.data.iter_mut().for_each(|x| *x += rhs)
    }
}

impl<F, const N: usize> MulAssign<F> for Vector<F, N>
where
    F: MulAssign<F> + Copy,
{
    fn mul_assign(&mut self, rhs: F) {
        self.data.iter_mut().for_each(|x| *x *= rhs)
    }
}

/// This is the corresponding matrix structure, instantiating matrices made me run into stack
/// overflows with nalgebra, whose static sized matrices aren't (yet?) designed for large
/// allocations.
#[derive(Debug, Clone)]
pub struct Matrix<F, const N: usize, const M: usize> {
    data: Box<[Vector<F, M>; N]>,
}

impl<F, const N: usize, const M: usize> Matrix<F, N, M>
where
    F: Copy,
{
    pub fn transpose(&self) -> Matrix<F, M, N> {
        let mut j = 0;
        let data = [(); M].map(|_| {
            let mut i = 0;
            let col = [(); N].map(|_| {
                let val = self.data[i].data[j];
                i += 1;
                val
            });
            j += 1;
            Vector {
                data: Box::new(col),
            }
        });
        Matrix {
            data: Box::new(data),
        }
    }

    pub fn from_distribution<Rng, Distribution>(rng: &mut Rng, dist: &Distribution) -> Self
    where
        Rng: rand::Rng,
        Distribution: rand_distr::Distribution<F>,
    {
        let data = [(); N].map(|_| Vector::<F, M>::from_distribution(rng, dist));
        Matrix {
            data: Box::new(data),
        }
    }
}

impl<F, const N: usize, const M: usize> Matrix<F, N, M>
where
    F: Zero + Copy,
{
    pub fn new() -> Self {
        let data = [(); N].map(|_| Vector::<F, M>::new());
        Matrix {
            data: Box::new(data),
        }
    }

    pub fn from_iter<I: Iterator<Item = F>>(mut iter: I) -> Self {
        let data = Box::new([(); N].map(|_| {
            let data = Box::new([(); M].map(|_| iter.next().unwrap_or(F::zero())));
            Vector { data }
        }));
        Matrix { data }
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.data.iter().flat_map(|row| row.data.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.data.iter_mut().flat_map(|row| row.data.iter_mut())
    }

    pub fn row_iter(&self) -> impl Iterator<Item = &Vector<F, M>> {
        self.data.iter()
    }

    pub fn row_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Vector<F, M>> {
        self.data.iter_mut()
    }
}

impl<F, const N: usize, const M: usize> Matrix<F, N, M>
where
    F: One + Copy,
{
    fn ones() -> Self {
        let data = [(); N].map(|_| Vector::<F, M>::ones());
        Matrix {
            data: Box::new(data),
        }
    }
}

impl<'a, F, const N: usize, const M: usize> Add<&'a Matrix<F, N, M>> for &'a Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Zero + Copy,
{
    type Output = Matrix<F, N, M>;

    fn add(self, rhs: &'a Matrix<F, N, M>) -> Self::Output {
        let Matrix { data: left_data } = self;
        let Matrix { data: right_data } = rhs;
        let mut row_iter = left_data
            .iter()
            .zip(right_data.iter())
            .map(|(left_row, right_row)| left_row + right_row);
        // This is guaranteed to not excetpp by the sizes
        let data = [(); N].map(move |_| row_iter.next().unwrap());
        Matrix {
            data: Box::new(data),
        }
    }
}

impl<F, const N: usize, const M: usize> Add<Matrix<F, N, M>> for Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Zero + Copy,
{
    type Output = Matrix<F, N, M>;

    fn add(self, rhs: Matrix<F, N, M>) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, F, const N: usize, const M: usize> Add<&'a Matrix<F, N, M>> for Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Zero + Copy,
{
    type Output = Matrix<F, N, M>;

    fn add(self, rhs: &Matrix<F, N, M>) -> Self::Output {
        &self + rhs
    }
}

impl<'a, F, const N: usize, const M: usize> Add<Matrix<F, N, M>> for &'a Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Zero + Copy,
{
    type Output = Matrix<F, N, M>;

    fn add(self, rhs: Matrix<F, N, M>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, F, const N: usize, const M: usize, const P: usize> Mul<&'a Matrix<F, M, P>>
    for Matrix<F, N, M>
where
    F: Copy + Add<F, Output = F> + Mul<F, Output = F> + Zero,
{
    type Output = Matrix<F, N, P>;

    /// This is inefficient.  It should be done with something like a transpose view, so we don't
    /// copy the matrix twice
    fn mul(self, rhs: &'a Matrix<F, M, P>) -> Self::Output {
        let transpose = rhs.transpose();
        let mut row_iter = self.data.iter().map(|row| {
            let mut cols_iter = transpose.data.iter().map(|row_for_col| row_for_col * row);
            let data = Box::new([(); P].map(|_| cols_iter.next().unwrap()));
            Vector { data }
        });
        let data = Box::new([(); N].map(|_| row_iter.next().unwrap()));
        Matrix { data }
    }
}

/// Here we implement the cross type multiplications
impl<'a, F, const N: usize, const M: usize> Mul<&'a Vector<F, M>> for &'a Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: &'a Vector<F, M>) -> Self::Output {
        let row_iter = self.data.iter().map(|row| row * rhs);
        Vector::from_iter(row_iter)
    }
}

impl<'a, F, const N: usize, const M: usize> Mul<Vector<F, M>> for &'a Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: Vector<F, M>) -> Self::Output {
        self * &rhs
    }
}

impl<'a, F, const N: usize, const M: usize> Mul<&'a Vector<F, M>> for Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: &'a Vector<F, M>) -> Self::Output {
        &self * rhs
    }
}

impl<F, const N: usize, const M: usize> Mul<Vector<F, M>> for Matrix<F, N, M>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + Copy + Zero,
{
    type Output = Vector<F, N>;

    fn mul(self, rhs: Vector<F, M>) -> Self::Output {
        &self * &rhs
    }
}

impl<F, const N: usize, const M: usize> Mul<&Matrix<F, N, M>> for &Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + AddAssign<F> + Copy + Zero,
{
    type Output = Vector<F, M>;
    fn mul(self, rhs: &Matrix<F, N, M>) -> Vector<F, M> {
        let mut accum = Vector::<F, M>::new();
        self.iter()
            .zip(rhs.row_iter())
            .for_each(|(vec_entry, matrix_row)| {
                accum += matrix_row * *vec_entry;
            });

        accum
    }
}

impl<F, const N: usize, const M: usize> Mul<Matrix<F, N, M>> for &Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + AddAssign<F> + Copy + Zero,
{
    type Output = Vector<F, M>;
    fn mul(self, rhs: Matrix<F, N, M>) -> Vector<F, M> {
        self * &rhs
    }
}

impl<F, const N: usize, const M: usize> Mul<&Matrix<F, N, M>> for Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + AddAssign<F> + Copy + Zero,
{
    type Output = Vector<F, M>;
    fn mul(self, rhs: &Matrix<F, N, M>) -> Vector<F, M> {
        &self * rhs
    }
}

impl<F, const N: usize, const M: usize> Mul<Matrix<F, N, M>> for Vector<F, N>
where
    F: Add<F, Output = F> + Mul<F, Output = F> + AddAssign<F> + Copy + Zero,
{
    type Output = Vector<F, M>;
    fn mul(self, rhs: Matrix<F, N, M>) -> Vector<F, M> {
        &self * &rhs
    }
}

#[cfg(test)]
mod test {
    const TOL: f64 = 1.0e-7;

    use crate::mat_mul::{Matrix, Vector};
    use num_traits::Zero;
    use rand::thread_rng;
    use rand_distr::{self, Distribution};

    #[test]
    fn test_transpose() {}

    #[test]
    fn test_vec_add() {
        const DIM: usize = 1000;

        let mut rng = rand::thread_rng();
        let x =
            Vector::<f64, 1000>::from_distribution(&mut rng, &rand_distr::Uniform::new(-1.0, 1.0));
        let y =
            Vector::<f64, 1000>::from_distribution(&mut rng, &rand_distr::Uniform::new(-1.0, 1.0));

        let z = &x + &y;

        x.data
            .iter()
            .zip(y.data.iter())
            .zip(z.data.iter())
            .for_each(|((x_el, y_el), z_el)| {
                assert!((*x_el + *y_el - *z_el).abs() <= TOL);
            });
    }

    #[test]
    fn test_map() {
        const DIM: usize = 20;
        const LIMIT: usize = 10;
        let iter_vec = Vector::<i32, DIM>::from_iter((0..LIMIT).map(|x| x as i32));
        let mut map_vec = Vector::<i32, DIM>::new();
        let mut count = 0;
        map_vec.map_inplace(|_| {
            let new_count = if count < LIMIT { count as i32 } else { 0 };
            count += 1;
            new_count
        });
        assert_eq!(iter_vec, map_vec)
    }

    #[test]
    fn test_add_and_scalar_mult() {
        const DIM: usize = 200;
        let ones = Vector::<i32, DIM>::ones();
        let mut neg = ones.clone();
        neg *= -1;
        let zeros = Vector::<i32, DIM>::new();
        assert_eq!(&ones + &neg, zeros);
    }

    #[test]
    fn test_transpose_mul() {
        const ROWS: usize = 2;
        const COLS: usize = 3;
        let row_1 = [1, 2, 3];
        let row_2 = [0, 4, 5];

        let data = [row_1, row_2];
        let matrix = Matrix::<i32, ROWS, COLS>::from_iter(
            data.iter().flat_map(|row| row.iter()).map(|x| *x),
        );
        let row_1 = Vector::<i32, COLS>::from_iter(row_1.iter().map(|x| *x));
        let row_2 = Vector::<i32, COLS>::from_iter(row_2.iter().map(|x| *x));

        let upper_vector = Vector::<i32, ROWS>::from_iter([1, 0].iter().map(|x| *x));
        let lower_vector = Vector::<i32, ROWS>::from_iter([0, 1].iter().map(|x| *x));
        assert_eq!(&upper_vector * &matrix, row_1);
        assert_eq!(&lower_vector * &matrix, row_2);
        assert_eq!(
            ((&lower_vector * 3) + (&upper_vector * 2)) * matrix,
            (&row_2 * 3) + (&row_1 * 2)
        );

        const BIG_ROWS: usize = 100;
        const BIG_COLS: usize = 200;

        let mut rng = thread_rng();
        let poisson_pi = rand_distr::Poisson::new(3.1415926)
            .unwrap()
            .map(|x| x as i32);
        let left = Vector::<i32, BIG_ROWS>::from_distribution(&mut rng, &poisson_pi);
        let right = Vector::<i32, BIG_COLS>::from_distribution(&mut rng, &poisson_pi);
        let matrix = Matrix::<i32, BIG_ROWS, BIG_COLS>::from_distribution(&mut rng, &poisson_pi);
        assert_eq!((&left * &matrix) * &right, &left * (&matrix * &right));
    }
}
