##  Detecting hidden but non-trivial problems in transfer learning models using Amazon SageMaker Debugger

This [notebook](https://github.com/aws-samples/amazon-sagemaker-debug-ml-model-training/blob/main/debug_ml_model_training.ipynb) contains the notebook and training scripts for the blogpost **Detecting hidden but non-trivial problems in transfer learning models using Amazon SageMaker Debugger**

In the notebok, we’ll show you an end-to-end example of doing transfer learning and using SageMaker Debugger to detect hidden problems that can cause serious consequences that would not have been easily uncovered. Debugger does not incur additional cost if you are running training on Sagemaker. Moreover, you can enable the [built-in rules](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html) with just a couple lines of code when you call the Sagemaker estimator function. Our task is to do transfer learning using a ResNet Model to recognize [German traffic sign dataset](https://ieeexplore.ieee.org/document/6033395).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

