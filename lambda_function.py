import os
import datetime

import boto3

aws_access_key_id = os.getenv('aws_access_key_id')
aws_default_region = os.getenv('aws_default_region')
aws_secret_access_key = os.getenv('aws_secret_access_key')

client = boto3.client('sagemaker',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)


def lambda_handler(event, context):
    '''
    {
        'model': 'dpr' | 'debarta',
        'container_entrypoint': ['/opt/ml/dpr_training/run.sh', 'small'|'long'] | ['/opt/ml/debarta_training/run.sh', 'small'|'long_1'|'long_2', 512|1024|1024, 256|512|512]
    }
    '''
    model_name = event['model']
    if model_name in ('dpr', 'debarta'):
        container_entrypoint = event['container_entrypoint']
        model_file_name = f'{model_name}_{container_entrypoint[1]}.zip'
        train_data_file = f'{model_name}_{container_entrypoint[1]}_train.json'
        test_data_file = f'{model_name}_{container_entrypoint[1]}_test.json'
    else:
        raise NameError(f"`{model_name}` is not a valid model name.")

    current_timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')

    environment_vars = {
        'dpr': {
            'n_epochs': os.getenv('dpr_n_epochs', '30'),
            'batch_size': os.getenv('dpr_batch_size', '8'),
            'learning_rate': os.getenv('dpr_learning_rate', '1e-5'),
            'weight_decay': os.getenv('dpr_weight_decay', '0.1'),
            'grad_acc_steps': os.getenv('dpr_grad_acc_steps', '4'),
            'evaluate_every': os.getenv('dpr_evaluate_every', '500')
        },
        'debarta': {}
    }

    response = trigger_training(current_timestamp, container_entrypoint, model_file_name, train_data_file, test_data_file, environment_vars[model_name])
    return {
        'statusCode': 200,
        'body': response
    }


def trigger_training(current_timestamp, container_entrypoint, model_file_name, train_data_file, test_data_file, environment_vars):
    response = client.create_training_job(
        TrainingJobName=f'clm-training-job-{current_timestamp}',
        AlgorithmSpecification={
            'TrainingImage': os.getenv('training_image'),
            'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['/bin/bash'],
            'ContainerArguments': container_entrypoint,
        },
        RoleArn=os.getenv('role_arn'),
        ResourceConfig={
            'InstanceType': os.getenv('instance_type'),
            # 'InstanceType': 'ml.g4dn.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        },
        InputDataConfig=[
            {
                'ChannelName': 'model',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://clm-artifacts/training_data/{model_file_name}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'InputMode': 'File'
            },
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://clm-artifacts/training_data/{train_data_file}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'InputMode': 'File'
            },
            {
                'ChannelName': 'test',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://clm-artifacts/training_data/{test_data_file}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'InputMode': 'File'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': 's3://clm-artifacts/models',
            'CompressionType': 'NONE'
        },
        Environment=environment_vars,
        Tags=[
            {
                'Key': 'Name',
                'Value': 'CLM Model Training'
            },
            {
                'Key': 'Project',
                'Value': 'CLM'
            },
        ]
    )
    return response
