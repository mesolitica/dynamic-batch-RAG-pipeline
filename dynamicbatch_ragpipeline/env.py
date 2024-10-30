import argparse
import logging
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--reload', type=lambda x: x.lower() == 'true',
        default=os.environ.get('reload', 'false').lower() == 'true',
        help='Enable hot loading (default: %(default)s, env: RELOAD)'
    )
    parser.add_argument(
        '--model-doc-layout',
        default=os.environ.get(
            'MODEL_DOC_LAYOUT',
            'yolo10'),
        help='Model type (default: %(default)s, env: MODEL_DOC_LAYOUT)'
    )
    parser.add_argument(
        '--dynamic-batching', type=lambda x: x.lower() == 'true',
        default=os.environ.get('DYNAMIC_BATCHING', 'true').lower() == 'true',
        help='Enable dynamic batching (default: %(default)s, env: DYNAMIC_BATCHING)'
    )
    parser.add_argument(
        '--dynamic-batching-microsleep', type=float,
        default=float(os.environ.get('DYNAMIC_BATCHING_MICROSLEEP', '1e-4')),
        help='microsleep to group dynamic batching, 1 / 1e-4 = 10k steps for second (default: %(default)s, env: DYNAMIC_BATCHING_MICROSLEEP)'
    )
    parser.add_argument(
        '--dynamic-batching-batch-size', type=float,
        default=int(os.environ.get('DYNAMIC_BATCHING_BATCH_SIZE', '20')),
        help='maximum of batch size during dynamic batching (default: %(default)s, env: DYNAMIC_BATCHING_BATCH_SIZE)'
    )
    parser.add_argument(
        '--accelerator-type', default=os.environ.get('ACCELERATOR_TYPE', 'cuda'),
        help='Accelerator type (default: %(default)s, env: ACCELERATOR_TYPE)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=int(
            os.environ.get(
                'MAX_CONCURRENT',
                '100')),
        help='Maximum concurrent requests (default: %(default)s, env: MAX_CONCURRENT)'
    )
    parser.add_argument(
        '--hotload', type=lambda x: x.lower() == 'true',
        default=os.environ.get('HOTLOAD', 'true').lower() == 'true',
        help='Enable hot loading (default: %(default)s, env: HOTLOAD)'
    )

    args = parser.parse_args()

    if args.model_doc_layout not in {'yolo10'}:
        raise ValueError('Currently document layout, `--model-doc-layout` or `MODEL_DOC_LAYOUT` environment variable, only support YOLO10.')

    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

logging.info(f'Serving app using {args}')