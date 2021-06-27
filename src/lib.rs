#![feature(array_map)]
use crossbeam;
use num_traits::One;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use thiserror::Error;
pub mod mat_mul;
pub struct PredictiveCodingNodeData<Float> {
    node_val: Float,
    error_val: Float,
}

pub struct InputWithErrorBackProp<Input> {
    pub forward_data: crossbeam::channel::Receiver<Input>,
    pub reverse_error_prop: crossbeam::channel::Sender<Input>,
}

pub struct OutputWithErrorBackProp<Output> {
    pub forward_data: crossbeam::channel::Sender<Output>,
    pub reverse_error_prop: crossbeam::channel::Receiver<Output>,
}

pub struct RootNode<RootInput, Output> {
    // pub node_err: Output,
    pub forward_input: crossbeam::channel::Receiver<RootInput>,
    pub forward_output: OutputWithErrorBackProp<Output>,
}

pub struct ForwardOwned<Input, Output> {
    input: Input,
    activation: Output,
    prediction_error: Input,
}

pub struct BackwardOwned<Calculator, Input> {
    calculator: Calculator,
    prediction: Input,
}

#[derive(Debug, Error)]
pub enum PreCodError<Output: std::fmt::Debug> {
    #[error("Unable to obtain a lock because it is poisoned")]
    PoisonedMutex,
    #[error("Unable to send data to channel: {0:?}")]
    SendToNext(Output),
}

impl<S: std::fmt::Debug, T> From<std::sync::PoisonError<T>> for PreCodError<S> {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        PreCodError::PoisonedMutex
    }
}

pub trait PredictiveCoding<F> {
    type InputType;
    type OutputType;
    fn smooth_input(&self, old_input: &mut Self::InputType, new_input: &Self::InputType);
    fn smooth_activation(
        &self,
        old_activation: &mut Self::OutputType,
        new_activation: &Self::OutputType,
    );

    fn set_prediction_error(
        &self,
        old_prediction_errors: &mut Self::InputType,
        predictions: &Self::InputType,
        input: &Self::InputType,
    );
}

pub trait DiffComp {
    type InputType;
    type OutputType;

    fn forward(&self, input: &Self::InputType) -> Self::OutputType;

    fn backward(
        &self,
        gradient: &Self::OutputType,
        activation: &Self::OutputType,
    ) -> Self::InputType;

    fn update(
        &mut self,
        input: &Self::InputType,
        error_from_next: &Self::OutputType,
        activation: &Self::OutputType,
        prediction_error: &Self::InputType,
        prediction: &mut Self::InputType,
    );
}

/// This exponential averaging does retains a bit over time, so if we increase the retention to 1
/// then we keep more of the old and add less of the new. This is used to keep more of the
/// historical activity of an activation or input
/// ```
/// # use precod_compdag::exp_avg;
/// # use precod_compdag::mat_mul::Vector;
/// # use rand::thread_rng;
/// # use rand_distr::StandardNormal;
/// # let mut rng = thread_rng();
/// # let dist = StandardNormal;
/// # const CENT: usize = 10;
/// # const TOL: f64 = 1.0e-7;
/// let mut old = Vector::<f64, CENT>::from_distribution(&mut rng, &dist);
/// let old_copy = old.clone();
/// let new = Vector::<f64, CENT>::from_distribution(&mut rng, &dist);
/// let zero = Vector::<f64, CENT>::new();
/// let smoothing = 0.25;
/// let retention = 0.75;
/// exp_avg(&retention, &mut old, &new);
/// let l1_norm = (&old + (((&old_copy * retention) + (&new * smoothing)) * -1.0))
/// .iter()
/// .map(|x| x.abs())
/// .sum::<f64>();
/// assert!( l1_norm <= TOL
/// );
/// let l1_norm = old.iter().map(|x| x.abs()).sum::<f64>();
/// exp_avg(&retention, &mut old, &zero);
/// assert!(l1_norm * retention - old.iter().map(|x| x.abs()).sum::<f64>() <= TOL)
/// ```
pub fn exp_avg<'a, F, SmoothingType>(
    retention: &F,
    old: &'a mut SmoothingType,
    new: &'a SmoothingType,
) where
    F: One + Div<F, Output = F> + Sub<F, Output = F> + Copy + Debug,
    SmoothingType: AddAssign<&'a SmoothingType> + MulAssign<F>,
{
    let one = <F as One>::one();
    let new_weight = one - *retention;
    println!("{:?}", new_weight);

    *old *= *retention / new_weight; // todo checked_div?
    *old += new; // Don't want to assign a new matrix by multiplying new_input
    *old *= new_weight;
}

pub trait ExpAvgPreCod<F> {
    type InputType: for<'a> AddAssign<&'a Self::InputType>
        + for<'a> SubAssign<&'a Self::InputType>
        + MulAssign<F>
        + Clone;
    type OutputType: for<'a> AddAssign<&'a Self::OutputType> + MulAssign<F>;
    const INPUT_RETENTION: F;
    const ACTIVATION_RETENTION: F;
}

impl<F, Calculator> PredictiveCoding<F> for Calculator
where
    Calculator: ExpAvgPreCod<F>,
    F: Div<F, Output = F> + One + Sub<F, Output = F> + Copy + Debug,
{
    type InputType = <Self as ExpAvgPreCod<F>>::InputType;

    type OutputType = <Self as ExpAvgPreCod<F>>::OutputType;

    fn smooth_input(&self, old_input: &mut Self::InputType, new_input: &Self::InputType) {
        exp_avg(&Self::INPUT_RETENTION, old_input, new_input)
    }

    fn smooth_activation(
        &self,
        old_activation: &mut Self::OutputType,
        new_activation: &Self::OutputType,
    ) {
        exp_avg(&Self::ACTIVATION_RETENTION, old_activation, new_activation)
    }

    fn set_prediction_error(
        &self,
        old_prediction_errors: &mut Self::InputType,
        predictions: &Self::InputType,
        input: &Self::InputType,
    ) {
        let mut new_prediction_error = (*predictions).clone();
        new_prediction_error -= input;
        *old_prediction_errors = new_prediction_error;
    }
}

pub struct BranchNode<F, Input, Output, RawInput, RawOutput, Calculator> {
    forward_owned: ForwardOwned<RawInput, RawOutput>,
    backward_owned: BackwardOwned<Calculator, RawInput>,
    pub input_channels: InputWithErrorBackProp<Input>,
    pub output_channels: OutputWithErrorBackProp<Output>,
    _pd: PhantomData<F>,
}

struct ForwardThreadData<F, Input, Output, RawInput, RawOutput, Calculator> {
    backprop_to_prev: crossbeam::channel::Sender<Input>,
    forward_data_to_next: crossbeam::channel::Sender<Output>,
    forward_data_from_prev: crossbeam::channel::Receiver<Input>,
    forward_owned_data: Arc<Mutex<ForwardOwned<RawInput, RawOutput>>>,
    read_backward_data_for_fore: Arc<Mutex<BackwardOwned<Calculator, RawInput>>>,
    _pd: PhantomData<F>,
}

impl<F, Input, Output, RawInput, RawOutput, Calculator>
    ForwardThreadData<F, Input, Output, RawInput, RawOutput, Calculator>
where
    Calculator: DiffComp<InputType = RawInput, OutputType = RawOutput>
        + PredictiveCoding<F, InputType = RawInput, OutputType = RawOutput>,
    Input: std::marker::Send + DataWrapper<RawInput> + 'static,
    Output: std::marker::Send + DataWrapper<RawOutput> + std::fmt::Debug + 'static,
{
    fn run(self) -> Result<(), PreCodError<Output>> {
        self.forward_data_from_prev
            .iter()
            .map(|input_from_prev| -> Result<(), PreCodError<Output>> {
                let ForwardOwned {
                    input,
                    activation,
                    prediction_error,
                } = &mut *self.forward_owned_data.lock()?;
                let BackwardOwned {
                    calculator,
                    prediction,
                } = &*self.read_backward_data_for_fore.lock()?;
                input_from_prev.data();
                calculator.smooth_input(input, input_from_prev.data());
                calculator.set_prediction_error(prediction_error, prediction, input);
                let output = calculator.forward(input);
                calculator.smooth_activation(activation, &output);
                self.forward_data_to_next
                    .send(Output::from_data(&output))
                    .map_err(|crossbeam::channel::SendError(output)| {
                        PreCodError::SendToNext(output)
                    })?;
                Ok(())
            })
            .collect::<Result<Vec<()>, PreCodError<Output>>>()
            .map(|_| ())
    }
}

struct BackPropThreadData<F, Output, RawInput, RawOutput, Calculator: DiffComp> {
    backprop_from_next: crossbeam::channel::Receiver<Output>,
    backward_owned_data: Arc<Mutex<BackwardOwned<Calculator, RawInput>>>,
    read_forward_data_for_back: Arc<Mutex<ForwardOwned<RawInput, RawOutput>>>,
    _pd: PhantomData<F>,
}

impl<F, Output, RawInput, RawOutput, Calculator>
    BackPropThreadData<F, Output, RawInput, RawOutput, Calculator>
where
    Calculator: DiffComp<InputType = RawInput, OutputType = RawOutput>
        + PredictiveCoding<F, InputType = RawInput, OutputType = RawOutput>,
    Output: std::marker::Send + DataWrapper<RawOutput> + 'static,
{
    fn run(&mut self) -> Result<(), PreCodError<()>> {
        let BackPropThreadData {
            backward_owned_data,
            read_forward_data_for_back,
            backprop_from_next,
            _pd,
        } = self;
        backprop_from_next
            .iter()
            .map(|error_from_next| -> Result<(), PreCodError<()>> {
                let ForwardOwned {
                    input,
                    activation,
                    prediction_error,
                } = &*read_forward_data_for_back.lock()?;
                let BackwardOwned {
                    calculator,
                    prediction,
                } = &mut *backward_owned_data.lock()?;

                calculator.update(
                    input,
                    error_from_next.data(),
                    activation,
                    prediction_error,
                    prediction,
                );
                Ok(())
            })
            .collect::<Result<Vec<()>, PreCodError<()>>>()
            .map(|_| ())
    }
}

impl<F, Input, Output, RawInput, RawOutput, Calculator>
    BranchNode<F, Input, Output, RawInput, RawOutput, Calculator>
where
    F: Copy + std::marker::Send + num_traits::Zero + 'static,
    Calculator: DiffComp<InputType = RawInput, OutputType = RawOutput>
        + PredictiveCoding<F, InputType = RawInput, OutputType = RawOutput>
        + std::marker::Send
        + 'static,
    RawOutput: Mul<F, Output = RawOutput> + std::marker::Send + 'static + Clone,
    RawInput: Mul<F, Output = RawInput> + std::marker::Send + 'static + Clone,
    Input: std::marker::Send + DataWrapper<RawInput> + 'static,
    Output: std::marker::Send + DataWrapper<RawOutput> + std::fmt::Debug + 'static,
{
    pub fn new(
        input_forward: crossbeam::channel::Receiver<Input>,
        input_error: crossbeam::channel::Sender<Input>,
        output_forward: crossbeam::channel::Sender<Output>,
        output_error: crossbeam::channel::Receiver<Output>,
        input_init: Input,
        activation_init: Output,
        calculator: Calculator,
    ) -> BranchNode<F, Input, Output, RawInput, RawOutput, Calculator> {
        let input_channels = InputWithErrorBackProp {
            forward_data: input_forward,
            reverse_error_prop: input_error,
        };

        let output_channels = OutputWithErrorBackProp {
            forward_data: output_forward,
            reverse_error_prop: output_error,
        };

        let input = input_init.data().clone();
        let prediction_error = input_init.data().clone() * F::zero();
        let forward_owned = ForwardOwned {
            input,
            activation: activation_init.data().clone(),
            prediction_error,
        };

        let backward_owned = BackwardOwned {
            calculator,
            prediction: input_init.data().clone(),
        };

        BranchNode {
            forward_owned,
            backward_owned,
            input_channels,
            output_channels,
            _pd: PhantomData,
        }
    }

    // this should probably not be pub, and should instead be called by whatever starts the whole
    // graph
    pub fn run<'a>(self) -> std::thread::JoinHandle<()> {
        let BranchNode {
            forward_owned,
            backward_owned,
            input_channels,
            output_channels,
            _pd,
        } = self;

        thread::spawn(move || {
            let forward_owned_data = Arc::new(Mutex::new(forward_owned));
            let read_forward_data_for_back = Arc::clone(&forward_owned_data);
            let InputWithErrorBackProp {
                forward_data: forward_data_from_prev,
                reverse_error_prop: backprop_to_prev,
            } = input_channels;
            let OutputWithErrorBackProp {
                forward_data: forward_data_to_next,
                reverse_error_prop,
            } = output_channels;

            let backward_owned_data = Arc::new(Mutex::new(backward_owned));
            let read_backward_data_for_fore = Arc::clone(&backward_owned_data);
            // backprop thread
            thread::spawn(move || {
                let mut back_prop_thread_data = BackPropThreadData {
                    backward_owned_data,
                    read_forward_data_for_back,
                    backprop_from_next: reverse_error_prop,
                    _pd,
                };

                back_prop_thread_data.run()
            });

            // forward thread
            thread::spawn(move || {
                let forward_thread_data = ForwardThreadData {
                    backprop_to_prev,
                    forward_data_to_next,
                    forward_data_from_prev,
                    forward_owned_data,
                    read_backward_data_for_fore,
                    _pd,
                };

                forward_thread_data.run(); // todo do somethign with the result
            });
        })
    }
}

pub struct LeafNode<Input, Labels> {
    pub node_data: Input,
    // pub node_mus: Input,
    pub input_channels: InputWithErrorBackProp<Input>,
    pub ground_truth_channel: crossbeam::channel::Receiver<Labels>,
}

pub trait DataWrapper<Data> {
    fn from_data(data: &Data) -> Self;
    fn data(&self) -> &Data;
    fn data_mut(&mut self) -> &mut Data;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
