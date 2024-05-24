from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
)


class CdkDeployLambdaStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Defines an AWS Lambda resource
        lambda_function = _lambda.Function(
            self, 'test',
            runtime=_lambda.Runtime.PYTHON_3_10,
            code=_lambda.Code.from_asset('./'),
            handler='lambda_function.lambda_handler',
        )


if __name__ == '__main__':
    app = cdk.App()
    CdkDeployLambdaStack(app, 'CdkDeployLambdaStack')
    app.synth()
