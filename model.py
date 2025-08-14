from neuralforecast.models import TimesNet, NHITS, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MAE, MAPE, MSE
import argparse


def get_time_series_model(model_name: str, **kwargs):
    """
    Factory function to create a time series model instance.
    """
    if model_name == 'TimesNet':
        return TimesNet(**kwargs)
    elif model_name == 'NHITS':
        return NHITS(**kwargs)
    elif model_name == 'PatchTST':
        return PatchTST(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds model-specific arguments to an ArgumentParser.
    """
    # General arguments
    parser.add_argument('--model_name', type=str, required=True, choices=['TimesNet', 'NHITS', 'PatchTST'],
                        help='Name of the model to create.')
    parser.add_argument('--h', type=int, default=24, help='Forecast horizon.')
    parser.add_argument('--input_size', type=int, default=72, help='Input window size (look-back).')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3, dest='learning_rate',
                        help='Learning rate for the optimizer.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of training steps.')

    parser.add_argument('--loss', type=str, default='DistributionLoss',
                        choices=['DistributionLoss', 'MAPE', 'MAE', 'MSE'],
                        help='Loss function to use for training.')
    # --- Model-specific arguments ---
    # For TimesNet
    parser.add_argument('--hidden_size', type=int, default=64, help='(TimesNet) Hidden size.')
    parser.add_argument('--conv_hidden_size', type=int, default=64, help='(TimesNet) Convolutional hidden size.')

    # For PatchTST
    parser.add_argument('--patch_len', type=int, default=16, help='(PatchTST) Patch length.')
    parser.add_argument('--stride', type=int, default=8, help='(PatchTST) Stride.')

    # For NHITS
    parser.add_argument('--n_blocks', type=int, default=1, help='(NHITS) Number of blocks per stack. A list of 3 will be created.')
    parser.add_argument('--mlp_units', type=int, nargs=2, default=[512, 512],
                        help='(NHITS) MLP units for each block, e.g., --mlp_units 512 512. A list of lists will be created.')
    
    return parser

def create_model_from_args(args: argparse.Namespace) -> object:
    """Creates a NeuralForecast model instance from parsed arguments."""
    LOSS_MAP = {
        'DistributionLoss': DistributionLoss(distribution='Normal', level=[80, 90]),
        'MAPE': MAPE(),
        'MAE': MAE(),
        'MSE': MSE()
    }
    # Instantiate the loss function based on the argument
    loss_function = LOSS_MAP.get(args.loss, MAE())

    # Prepare the parameters for the factory function
    model_params = {
        'h': args.h,
        'input_size': args.input_size,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'loss': loss_function,
        'scaler_type': 'standard',
        'val_check_steps': 50,
        'early_stop_patience_steps': 3
    }

    # Add model-specific parameters based on the chosen model
    if args.model_name == 'TimesNet':
        model_params['hidden_size'] = args.hidden_size
        model_params['conv_hidden_size'] = args.conv_hidden_size
    elif args.model_name == 'PatchTST':
        model_params['patch_len'] = args.patch_len
        model_params['stride'] = args.stride
    elif args.model_name == 'NHITS':
        # Construct lists for NHITS based on command-line arguments
        model_params['n_blocks'] = [args.n_blocks] * 3
        model_params['mlp_units'] = [args.mlp_units] * 3
        model_params['stack_types'] = ['identity'] * 3 # Common setting for general forecasting

    model_instance = get_time_series_model(model_name=args.model_name, **model_params)

    return model_instance

def main(args_list: list = None):
    """
    Main function to parse command-line arguments and create a model.
    This allows for testing model creation from the command line.

    Args:
        args_list (list, optional): A list of command-line arguments.
                                    If None, arguments are parsed from sys.argv.
                                    Defaults to None.

    Returns:
        An instantiated NeuralForecast model.
    """
    parser = argparse.ArgumentParser(
        description="Create and inspect a time series forecasting model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser = add_model_args(parser)
    args = parser.parse_args(args_list)
    model_instance = create_model_from_args(args)

    # Only print details when run as a standalone script
    if args_list is None:
        print(f"Creating model: {args.model_name}...")
        print(f"\nSuccessfully created '{args.model_name}' model instance.")
        print(f"Total trainable parameters: {count_parameters(model_instance):,}")
        print("\nModel configuration:")
        print(model_instance)

    return model_instance

if __name__ == '__main__':
    # This allows the script to be run from the command line as before.
    main()