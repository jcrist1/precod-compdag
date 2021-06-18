#![feature(array_map)]
use crossbeam;
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

pub struct BranchNode<Input, Output> {
    // pub node_mus: Input,
    // pub node_err: Output,
    pub input_channels: InputWithErrorBackProp<Input>,
    pub output_channels: OutputWithErrorBackProp<Output>,
}

pub struct LeafNode<Input, Labels> {
    pub node_data: Input,
    // pub node_mus: Input,
    pub input_channels: InputWithErrorBackProp<Input>,
    pub ground_truth_channel: crossbeam::channel::Receiver<Labels>,
}

//pub struct BranchNode<Float, Input, Output, ErrorInput, ErrorOutput> {
//    data: PredictiveCodingNodeData<Float>,
//    forward_input: crossbeam::channel::Receiver<RootInput>,
//    forward_output
//}
//
//pub struct Branch<Float
//pub enum PredictiveCodingNode<Float> {
//    Root {
//    },
//    Branch {},
//    Leaf {
//        output_channel:
//    },
//}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
