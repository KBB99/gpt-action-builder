from aws_cdk import (
    aws_lambda as _lambda,
    aws_iam as iam,
    core
)

class LambdaStack(core.Stack):

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        with open('get_data.py', encoding='utf8') as fp:
            handler_code = fp.read()

        lambda_function = _lambda.Function(
            self, 'LambdaHandler',
            code=_lambda.InlineCode(handler_code),
            handler='index.main',
            timeout=core.Duration.seconds(300),
            runtime=_lambda.Runtime.PYTHON_3_7,
        )

        lambda_function.add_to_role_policy(iam.PolicyStatement(
            actions=['dynamodb:GetItem'],
            resources=['arn:aws:dynamodb:*:*:table/SynthAgentStack-ConversationTable75C14D21-1V1ERMBWMZT4P']
        ))

app = core.App()
LambdaStack(app, 'LambdaStack')
app.synth()
