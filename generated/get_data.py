import boto3

def get_data_from_dynamodb(id):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('SynthAgentStack-ConversationTable75C14D21-1V1ERMBWMZT4P')
    response = table.get_item(Key={'SessionId': id})
    return response['Item']

print(get_data_from_dynamodb('us-east-1:f441973b-64c7-4ea8-ba4f-fc73d4951f94-0'))
