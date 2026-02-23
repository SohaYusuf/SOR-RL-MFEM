
import argparse
import pdb
import time
import pprint

from functions.utils import device
from functions.test import test_policy_fgmres, test_policy_sor


def argparser():
    """
    Parses command-line arguments for training parameters.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Training parameters for the model.")
    
    # Define training and configuration parameters 21 arguments
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Number of transitions sampled from the replay buffer")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor for future rewards")
    parser.add_argument("--eps_start", type=float, default=0.9, 
                        help="Starting value of epsilon for exploration")
    parser.add_argument("--eps_end", type=float, default=0.05, 
                        help="Final value of epsilon for exploration")
    parser.add_argument("--eps_decay", type=float, default=1000, 
                        help="Rate of epsilon decay (higher means slower decay)")
    parser.add_argument("--tau", type=float, default=0.005, 
                        help="Update rate of the target network")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate for the AdamW optimizer")
    parser.add_argument("--num_episodes", type=int, default=200, 
                        help="Total number of episodes for training")
    parser.add_argument("--target_tol", type=float, default=1e-10, 
                        help="target tolerance for flexible GMRES")
    parser.add_argument("--default_omega", type=float, default=1.3, 
                        help="default value of relaxation parameter" \
                        "for default_SOR e.g 1e-2")
    parser.add_argument("--omega_min", type=float, default=0.0, 
                        help="train or test")
    parser.add_argument("--omega_max", type=float, default=1.0, 
                        help="train or test")
    
    parser.add_argument("--max_iter", type=int, default=500, 
                        help="maximum number of FGMRES iterations")

    parser.add_argument("--debug", type=int, default=True, 
                        help="Debug flag")

    parser.add_argument("--train_data_path", type=str, default="data/advection/train/", 
                        help="Path to the dataset")
    parser.add_argument("--test_data_path", type=str, default="data/advection/test/", 
                        help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="results/", 
                        help="path where results will be saved")

    # parser.add_argument("--N", type=int, default=144, help="Parameter N")
    parser.add_argument("--dataset", type=str, default="advection", 
                        help="Dataset name e.g advection, advection")
    parser.add_argument("--checkpoint", type=str, 
                        help="checkpoint path that you want to use for evaluation")
    parser.add_argument("--mode", type=str, default="train", 
                        help="train or test")
    # parser.add_argument("--data_size_list", type=int, nargs="+", 
    #                     default=[3281], 
    #                     help="Use 144 576 etc as the argument, do not add comma") 
    parser.add_argument("--n_actions", type=int, default=50, 
                        help="number of discrete actions the RL agent can take")  

    parser.add_argument("--train_RL_model", type=int, default=True, 
                        help="Train RL agent with FGMRES")
    parser.add_argument("--run_FGMRES_baseline", type=int, default=True, 
                        help="Run FGMRES with no preconditioner")
    parser.add_argument("--run_FGMRES_default_SOR", type=int, default=True, 
                        help="Run FGMRES with default SOR")
    
    parser.add_argument("--run_sor_as_solver", type=int, default=False, 
                        help="Run SOR as solver (not preconditioner)")
    parser.add_argument("--run_fgmres_as_solver", type=int, default=True, 
                        help="Run FGMRES with SOR as preconditioner.")

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = argparser()
    
    # Logging
    print(f"Using device: {device}")
    print("Using config:")
    pprint.pprint(vars(args))
    print()

    if args.run_fgmres_as_solver==True:

        from functions.train_fgmres import train_fgmres
        if args.mode == "train":
            
            # train
            start_time = time.time()
            policy_net, all_residuals_dict = train_fgmres(vars(args))
            elapsed_time = time.time() - start_time
            print(f"Total training time: {elapsed_time:.2f} seconds")
    
            # test
            start_time = time.time()
            test_policy_fgmres(policy_net,
                vars(args))
           
            elapsed_time = time.time() - start_time
            print(f"Total testing time: {elapsed_time:.2f} seconds")

    if args.run_sor_as_solver==True:
         
        from functions.train_sor import train_sor
        if args.mode == "train":
            
            # this code works as of 10/27/2025 12:14 PM
            # train the SOR with fixed omega (check with theoretical omega for SPD system)
            start_time = time.time()
            policy_net, train_loader, all_residuals_dict = train_sor(vars(args))
            elapsed_time = time.time() - start_time
            print(f"Total training time: {elapsed_time:.2f} seconds")
            
            # test
            start_time = time.time()
            test_policy_sor(policy_net,
                train_loader,
                all_residuals_dict,
                vars(args))
           
            elapsed_time = time.time() - start_time
            print(f"Total testing time: {elapsed_time:.2f} seconds")

        elif args.mode =="test":
            # test the SOR with fixed omega (check with theoretical omega for SPD system)
            start_time = time.time()
            test_policy_sor(vars(args))
            elapsed_time = time.time() - start_time
            print(f"Total training time: {elapsed_time:.2f} seconds")

    elif args.mode == "test":
        # evaluate
        start_time = time.time()
        eval_(vars(args))
        elapsed_time = time.time() - start_time
        print(f"Total evaluation time: {elapsed_time:.2f} seconds")